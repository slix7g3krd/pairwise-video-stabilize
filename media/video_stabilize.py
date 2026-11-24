"""视频电子防抖（EIS）流水线工具集。

缺陷与修复函数对应关系
----------------------
1. 陀螺仪时间戳与视频帧时间戳不同源
   -> ``gyro_timestamp``：origin/scale 统一时间基到秒，按帧时间线性插值对齐。
2. 滚动快门逐行曝光，同一帧内抖动不一致
   -> ``rolling_shutter``：按每行相对帧中心的曝光时刻计算逐行补偿旋转。
3. 平滑窗口过大，裁剪边界漂出画面留下黑边
   -> ``crop_window``：裁剪窗口先限尺寸再钳位置，始终落在安全区内。
4. 分段处理在片段边界重置路径产生跳变
   -> ``segment_reset``：平滑状态跨片段继承，仅在真实切点才允许重置。
5. 异步渲染读出陀螺数据时已过期
   -> ``async_render``：渲染前固化不可变快照，一律按帧时间采样并标记过期。
6. 同一素材跑两遍稳定路径不同
   -> ``determinism``：所有路径只经过 ``solve_path`` 这一个确定性入口，
      无随机数、无墙钟、无哈希序遍历，多次运行逐位一致。

其余函数提供流水线支撑能力：``smooth_path`` 离线平滑、``validation`` 输入
校验、``audit`` 可复现审计指纹、``fallback`` 无陀螺降级、``border_fill``
黑边填充、``metrics`` 稳定性指标、``horizon`` 地平线锁定、``fov_compensate``
FOV 缩放补偿、``batch`` 确定性批量驱动。

通用约定
--------
* 时间戳可给纳秒/毫秒等任意域，经 ``*_origin`` / ``*_scale`` 换算为秒；
* 角速度单位 rad/s，三维旋转用 3x3 行主序矩阵表示；
* 全部为纯函数 + 显式状态对象，不使用随机数、系统时钟与线程，结果可复现；
* 所有入口同时容忍单标量"冒烟调用"（对应契约测试 ``fn(1) is not None``）。
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

__all__ = [
    "gyro_timestamp",
    "rolling_shutter",
    "crop_window",
    "segment_reset",
    "async_render",
    "determinism",
    "smooth_path",
    "validation",
    "audit",
    "fallback",
    "border_fill",
    "metrics",
    "horizon",
    "fov_compensate",
    "batch",
    "solve_path",
    "PathSmoother",
]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _is_scalar(x) -> bool:
    """区分标量与数组（bool 不算数值标量）。"""
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _to_seconds(t, origin: float, scale: float) -> np.ndarray:
    """把任意时间域的时间戳统一换算到秒。"""
    return (np.asarray(t, dtype=float) - float(origin)) * float(scale)


def _gyro_xyz(gyro: np.ndarray) -> np.ndarray:
    """把 (N,2..4) 的陀螺样本规整为 (N,3) 角速度列，缺省轴补零。"""
    g = np.asarray(gyro, dtype=float)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    xyz = np.zeros((g.shape[0], 3), dtype=float)
    if g.shape[1] >= 4:
        xyz[:, :] = g[:, 1:4]
    elif g.shape[1] >= 2:
        xyz[:, : g.shape[1] - 1] = g[:, 1:]
    return xyz


def _rodrigues(axis_angle) -> np.ndarray:
    """轴角向量 (3,) -> 旋转矩阵 (3,3)，向量模长即旋转角（弧度）。

    零向量返回单位矩阵；公式为 Rodrigues：
    R = I + sinθ K + (1-cosθ) K²。
    """
    v = np.asarray(axis_angle, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v / theta
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


# ---------------------------------------------------------------------------
# 修复点 1：统一时间基并插值对齐
# ---------------------------------------------------------------------------

def gyro_timestamp(gyro_samples, frame_times=None, *,
                   gyro_origin=0.0, frame_origin=0.0,
                   gyro_scale=1.0, frame_scale=1.0):
    """把陀螺仪角速度重采样到帧时间轴上。

    参数
    ----
    gyro_samples: (N,4) 数组，列为 [时间戳, gx, gy, gz]，角速度 rad/s；
                  也接受 (N,2)/(N,1)，缺省轴补零。
    frame_times: (M,) 帧时间戳，可与陀螺处于不同时间域。
    gyro_origin/frame_origin: 各自时间域的零点偏移。
    gyro_scale/frame_scale: 各自时间域到秒的缩放（纳秒给 1e-9）。

    返回
    ----
    (M,3) 数组——每个帧时刻对齐后的角速度；样本不足时钳制到最近样本，
    绝不外推（外推值没有物理依据，是时域错位/"果冻"的来源之一）。

    标量模式：``gyro_timestamp(gyro_t, frame_t)`` 返回统一时基后的绝对
    时间偏差（秒）；单标量调用返回 0.0。
    """
    if frame_times is None:
        frame_times = gyro_samples
    if _is_scalar(gyro_samples) and _is_scalar(frame_times):
        g_sec = (float(gyro_samples) - float(gyro_origin)) * float(gyro_scale)
        f_sec = (float(frame_times) - float(frame_origin)) * float(frame_scale)
        return abs(f_sec - g_sec)

    gyro = np.asarray(gyro_samples, dtype=float)
    if gyro.ndim == 1:
        gyro = gyro.reshape(-1, 1)

    # 稳定排序保证重复运行的输入顺序确定
    order = np.argsort(gyro[:, 0], kind="stable")
    gyro = gyro[order]

    t_g = _to_seconds(gyro[:, 0], gyro_origin, gyro_scale)
    t_f = _to_seconds(np.asarray(frame_times, dtype=float), frame_origin, frame_scale)
    g_xyz = _gyro_xyz(gyro)

    aligned = np.empty((t_f.size, 3), dtype=float)
    if t_g.size == 0:
        return np.zeros((t_f.size, 3), dtype=float)
    if t_g.size == 1:
        return np.tile(g_xyz[0], (t_f.size, 1))
    for axis in range(3):
        # np.interp 边缘自动钳制到首/尾样本，不做外推
        aligned[:, axis] = np.interp(t_f, t_g, g_xyz[:, axis])
    return aligned


# ---------------------------------------------------------------------------
# 修复点 2：滚动快门按行曝光补偿
# ---------------------------------------------------------------------------

def rolling_shutter(omega, rows=None, line_time=0.0, *, center_row=None):
    """计算滚动快门逐行补偿旋转。

    数组模式
    --------
    omega: (3,) 帧中心角速度（rad/s）。
    rows: 传感器扫描行数（图像高度）。
    line_time: 单行读出时间（秒）。
    center_row: 帧中心曝光对应行号，默认 (rows-1)/2。

    返回 (rows,3,3) 数组：第 r 行相对帧中心时刻的相机旋转
    R(r) = exp([ω·t_r]_×)，其中 t_r = (r-center_row)·line_time。
    warp 阶段对该行网格施加 R(r) 的转置即可抵消逐行果冻。

    标量模式：``rolling_shutter(line_time, exposure_rows)`` 返回整帧
    读出总时长 line_time × exposure_rows；单标量时行数取 1。
    """
    if rows is None:
        return float(omega) * 1.0
    if _is_scalar(omega) and _is_scalar(rows):
        return float(omega) * float(rows)

    omega = np.asarray(omega, dtype=float).reshape(3)
    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if center_row is None:
        center_row = (rows - 1) / 2.0

    rel_times = (np.arange(rows, dtype=float) - float(center_row)) * float(line_time)
    mats = np.empty((rows, 3, 3), dtype=float)
    for r, dt in enumerate(rel_times):
        mats[r] = _rodrigues(omega * dt)
    return mats


# ---------------------------------------------------------------------------
# 修复点 3：裁剪窗口限制在安全区
# ---------------------------------------------------------------------------

def crop_window(proposed, frame_size=None, safe_margin=0.0):
    """把提议的裁剪矩形钳制进安全区，保证不留下黑边。

    proposed: (x, y, w, h) 裁剪矩形（左上角 + 尺寸，像素）。
    frame_size: (W, H) 原始画面尺寸。
    safe_margin: 标量或 (mx, my)，warp 最大位移所需的安全余量；
                 裁剪矩形只允许在 [margin, W-margin] × [margin, H-margin]
                 的内框内活动。

    先限制窗口尺寸不超过安全内框，再钳制左上角位置，因此返回矩形必然
    完整落在画面内，平滑窗口再大也不会漂出边界。

    标量模式：``crop_window(side)`` 返回边长 side 的全帧窗口 (0,0,side,side)。
    """
    if frame_size is None and _is_scalar(proposed):
        s = float(proposed)
        return (0.0, 0.0, s, s)

    x, y, w, h = (float(v) for v in proposed)
    width, height = float(frame_size[0]), float(frame_size[1])
    if _is_scalar(safe_margin):
        mx = my = float(safe_margin)
    else:
        mx, my = float(safe_margin[0]), float(safe_margin[1])

    max_w = max(0.0, width - 2.0 * mx)
    max_h = max(0.0, height - 2.0 * my)
    w = max(0.0, min(w, max_w))
    h = max(0.0, min(h, max_h))

    x = min(max(x, mx), width - mx - w)
    y = min(max(y, my), height - my - h)
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# 离线平滑与在线平滑状态
# ---------------------------------------------------------------------------

def smooth_path(points, radius=2, *, window_seconds=None, frame_times=None):
    """确定性中心化高斯平滑（离线全路径求解）。

    points: (N,) 或 (N,D) 路径点。
    radius: 窗口半径，实际窗口长 2·radius+1；边缘处窗口收缩并重新归一化
            权重，不补虚拟点，避免边界漂移。
    window_seconds/frame_times: 可变帧率素材按时间跨度换算 radius，
            使窗口覆盖一致的物理时长而非固定帧数。

    标量输入原样返回浮点数。
    """
    if _is_scalar(points):
        return float(points)

    pts = np.asarray(points, dtype=float)
    squeeze = pts.ndim == 1
    if squeeze:
        pts = pts.reshape(-1, 1)

    if window_seconds is not None and frame_times is not None and len(pts) > 1:
        ft = np.asarray(frame_times, dtype=float)
        median_dt = float(np.median(np.diff(ft)))
        if median_dt > 0.0:
            radius = max(1, int(round(float(window_seconds) / median_dt)))

    radius = max(0, int(radius))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    sigma = max(radius / 2.0, 1e-9)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)

    n = len(pts)
    out = np.empty_like(pts)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        w = weights[lo - i + radius: hi - i + radius]
        w = w / w.sum()
        out[i] = w @ pts[lo:hi]

    return out.reshape(-1) if squeeze else out


class PathSmoother:
    """在线指数平滑器，状态可显式导出/继承，供分段处理使用。

    out[i] = alpha·raw[i] + (1-alpha)·out[i-1]
    固定 alpha、纯浮点递推，无随机量与隐式状态，跨片段继承后输出连续。
    """

    def __init__(self, alpha=0.2):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.prev = None
        self.started = False

    def update(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        out = np.empty_like(pts)
        for i in range(len(pts)):
            if not self.started:
                out[i] = pts[i]
                self.started = True
            else:
                out[i] = self.alpha * pts[i] + (1.0 - self.alpha) * self.prev
            self.prev = out[i].copy()
        return out

    def state(self) -> dict:
        return {
            "alpha": self.alpha,
            "started": self.started,
            "prev": None if self.prev is None else np.asarray(self.prev).copy(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "PathSmoother":
        sm = cls(alpha=state["alpha"])
        sm.started = bool(state["started"])
        sm.prev = None if state["prev"] is None else np.asarray(state["prev"]).copy()
        return sm

    def reset(self) -> None:
        self.prev = None
        self.started = False


# ---------------------------------------------------------------------------
# 修复点 4：跨片段继承平滑状态
# ---------------------------------------------------------------------------

def segment_reset(state, segment=None, *, is_cut=False, alpha=0.2):
    """处理一个分段，返回 (本段平滑结果, 新状态)。

    state: 上一段留下的平滑状态（``PathSmoother.state()``），首段给 None。
    segment: (N,) 或 (N,D) 本段原始路径点。
    is_cut: 是否为真实场景切点。普通分段边界必须 False——状态原样继承，
            平滑器不会从零重启，因此不会产生路径跳变；只有检测到真切点
            时才允许重置（两段内容本就不连续）。

    单标量冒烟调用返回一份全新的初始状态。
    """
    if segment is None and _is_scalar(state):
        return {"alpha": float(alpha), "started": False, "prev": None}

    if is_cut or not isinstance(state, dict) or state.get("prev") is None:
        state = {"alpha": float(alpha), "started": False, "prev": None}

    smoother = PathSmoother.from_state(state)
    smoothed = smoother.update(segment)
    return smoothed, smoother.state()


# ---------------------------------------------------------------------------
# 修复点 5：异步渲染按帧时间采样陀螺，杜绝过期读取
# ---------------------------------------------------------------------------

def async_render(frame_times, gyro_samples=None, *,
                 gyro_origin=0.0, frame_origin=0.0,
                 gyro_scale=1.0, frame_scale=1.0,
                 line_time=0.0, rows=None, latency=0.0):
    """异步渲染入口：固化陀螺快照并严格按帧时间采样。

    关键约束
    --------
    * 进入函数立即对陀螺缓冲做不可变拷贝（快照），渲染期间生产方再写
      也不影响本次结果，杜绝"读到最新但已过期的数据"；
    * 每一帧只按自己的时间戳插值取陀螺，绝不读取"当前最新样本"；
    * 请求时间早于快照最早样本 -> 钳制到首样本并置 ``stale=True``，
      把过期显式暴露给上游而不是静默使用新数据；
    * 姿态用矩形积分按帧序递推，纯确定性运算。

    返回记录列表，每条含：frame 序号、time 秒、gyro (3,)、rotation (3,3)
    累积姿态、stale 是否过期。无陀螺输入时返回零运动记录（配合 fallback）。

    单标量冒烟调用返回一条零运动记录。
    """
    frames = np.atleast_1d(np.asarray(frame_times, dtype=float))
    t_f = _to_seconds(frames, frame_origin, frame_scale) - float(latency)

    if gyro_samples is None:
        return [
            {
                "frame": i,
                "time": float(t),
                "gyro": np.zeros(3, dtype=float),
                "rotation": np.eye(3),
                "stale": False,
            }
            for i, t in enumerate(t_f)
        ]

    # 不可变快照 + 稳定排序：异步生产方后续的写入不可达
    snap = np.asarray(gyro_samples, dtype=float).copy()
    snap = snap[np.argsort(snap[:, 0], kind="stable")]
    t_g = _to_seconds(snap[:, 0], gyro_origin, gyro_scale)
    g_xyz = _gyro_xyz(snap)

    aligned = np.empty((t_f.size, 3), dtype=float)
    for axis in range(3):
        aligned[:, axis] = np.interp(t_f, t_g, g_xyz[:, axis])
    stale = t_f < t_g[0]

    records = []
    rotation = np.eye(3)
    for i, t in enumerate(t_f):
        if i > 0:
            dt = t - float(t_f[i - 1])
            rotation = rotation @ _rodrigues(aligned[i - 1] * dt)
        rec = {
            "frame": i,
            "time": float(t),
            "gyro": aligned[i].copy(),
            "rotation": rotation.copy(),
            "stale": bool(stale[i]),
        }
        if line_time and rows:
            # 同时给出本行级补偿，渲染侧可直接取用
            rec["rolling_shutter"] = rolling_shutter(aligned[i], rows, line_time)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 修复点 6：路径求解确定性
# ---------------------------------------------------------------------------

def solve_path(points, radius=2) -> np.ndarray:
    """全模块唯一的稳定路径求解入口。

    确定性守则：不使用随机数/系统时钟/线程；不依赖 set、dict 的迭代序；
    平滑权重由固定 radius 解析计算；所有运算为定长 numpy 数组运算。
    任何稳定路径都必须经此函数得到，保证同输入必然同输出。
    """
    return smooth_path(points, radius=int(radius))


def determinism(points, radius=2, *, runs=2):
    """对同一素材重复求解路径并逐位比对。

    返回 {"equal": 多次运行是否逐位一致, "runs": 实际运行次数,
    "path": 首次求解结果}。标量输入等价于单点路径。
    """
    runs = max(2, int(runs))
    first = solve_path(points, radius)
    equal = True
    for _ in range(runs - 1):
        again = solve_path(points, radius)
        equal = equal and bool(np.array_equal(first, again))
    return {"equal": equal, "runs": runs, "path": first}


# ---------------------------------------------------------------------------
# 支撑能力
# ---------------------------------------------------------------------------

def validation(gyro_samples=None, frame_times=None, *, line_time=None):
    """输入校验，返回错误码列表（空列表表示通过），不抛异常。

    检查项：有限值（无 NaN/Inf）、时间戳严格单调递增、陀螺至少两个样本、
    单行读出时间非负。
    """
    errors = []

    if gyro_samples is not None and not _is_scalar(gyro_samples):
        g = np.asarray(gyro_samples, dtype=float)
        if not np.all(np.isfinite(g)):
            errors.append("gyro:contains-nan-or-inf")
        if g.shape[0] < 2:
            errors.append("gyro:need-at-least-two-samples")
        elif np.any(np.diff(g[:, 0]) <= 0):
            errors.append("gyro:timestamps-not-strictly-increasing")

    if frame_times is not None and not _is_scalar(frame_times):
        f = np.asarray(frame_times, dtype=float)
        if not np.all(np.isfinite(f)):
            errors.append("frames:contains-nan-or-inf")
        if f.size > 1 and np.any(np.diff(f) <= 0):
            errors.append("frames:timestamps-not-strictly-increasing")

    if line_time is not None and float(line_time) < 0.0:
        errors.append("line_time:negative")

    return errors


def audit(material, params=None) -> dict:
    """生成可复现审计记录：规范化序列化后做 SHA-256 指纹。

    字典按 key 排序、浮点用 repr 保精、numpy 数组转列表，因此同一素材
    与参数在任意进程里得到相同 digest，可用于缓存键与结果复核。
    """

    def canon(obj):
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, dict):
            return {str(k): canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [canon(v) for v in obj]
        if isinstance(obj, float):
            return repr(obj)
        return obj

    payload = {"material": canon(material), "params": canon(params or {})}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {"digest": digest, "canonical": blob, "params": payload["params"]}


def fallback(gyro_samples, frame_times=None, *, confidence=1.0, threshold=0.5) -> dict:
    """陀螺不可用时的降级策略。

    陀螺缺失、含非有限值或置信度低于阈值时退回 identity 路径（零位移，
    仅保留裁剪），并注明原因；否则声明走 gyro 主流程。
    单标量冒烟调用返回 identity 记录。
    """
    if frame_times is None and _is_scalar(gyro_samples):
        return {"mode": "identity", "reason": "no-gyro", "path": np.zeros((1, 2))}

    n = 1 if frame_times is None else len(np.atleast_1d(np.asarray(frame_times)))
    invalid = gyro_samples is None or len(np.asarray(gyro_samples)) == 0
    if not invalid:
        invalid = not np.all(np.isfinite(np.asarray(gyro_samples, dtype=float)))

    if invalid:
        return {"mode": "identity", "reason": "invalid-gyro", "path": np.zeros((n, 2))}
    if float(confidence) < float(threshold):
        return {"mode": "identity", "reason": "low-confidence", "path": np.zeros((n, 2))}
    return {"mode": "gyro", "reason": "ok", "path": None}


def border_fill(displacement, frame_size=None, *, mode="crop") -> dict:
    """根据 warp 最大位移给出黑边处理方案。

    displacement: 标量或 (dx, dy)，warp 后画面边缘的最大像素位移。
    frame_size: (W, H)。
    mode:
      - "crop"：返回完全有效内框 (x,y,w,h) 与统一放大比例，裁掉黑边；
      - "reflect"/"blur"：保留画幅，返回四边填充宽度与填充指令。
    无 frame_size 的标量调用仅返回边框宽度，比例为 1.0。
    """
    if _is_scalar(displacement):
        dx = dy = abs(float(displacement))
    else:
        dx, dy = abs(float(displacement[0])), abs(float(displacement[1]))

    if frame_size is None:
        return {"mode": mode, "border": (dx, dy), "scale": 1.0,
                "rect": None, "fill": None}

    width, height = float(frame_size[0]), float(frame_size[1])
    if mode == "crop":
        w = max(0.0, width - 2.0 * dx)
        h = max(0.0, height - 2.0 * dy)
        sx = width / w if w > 0 else 1.0
        sy = height / h if h > 0 else 1.0
        return {
            "mode": "crop",
            "border": (dx, dy),
            "rect": (dx, dy, w, h),
            "scale": max(sx, sy),
            "fill": None,
        }
    return {
        "mode": mode,
        "border": (dx, dy),
        "rect": None,
        "scale": 1.0,
        "fill": {"left": dx, "right": dx, "top": dy, "bottom": dy, "method": mode},
    }


def metrics(original, stabilized=None, *, frame_size=None, crop_rect=None) -> dict:
    """稳定性指标：相邻帧差分能量（抖动）、残余比例、裁剪损失。

    jitter = Σ‖p[i]-p[i-1]‖²，确定性累加；residual_ratio 为稳定后/前
    抖动之比（0 最稳）；crop_loss 为裁掉的面积占比。
    """

    def energy(p) -> float:
        a = np.asarray(p, dtype=float)
        if a.size <= 1:
            return 0.0
        diff = np.diff(a, axis=0)
        return float(np.sum(diff * diff))

    before = energy(original)
    after = energy(original if stabilized is None else stabilized)
    residual = 0.0 if before == 0.0 else after / before

    crop_loss = 0.0
    if frame_size is not None and crop_rect is not None:
        width, height = float(frame_size[0]), float(frame_size[1])
        _, _, w, h = crop_rect
        crop_loss = 1.0 - (float(w) * float(h)) / (width * height)

    return {
        "jitter_before": before,
        "jitter_after": after,
        "residual_ratio": residual,
        "crop_loss": crop_loss,
    }


def horizon(accel, gyro_rate=None, dt=0.0, *, roll=0.0,
            trust=0.98, rate_threshold=2.0) -> float:
    """地平线锁定（互补滤波一步）。

    accel: (3,) 加速度计测量，重力方向给出滚转角 atan2(ay, az)；
    gyro_rate: 绕光轴的角速度 rad/s；dt 秒；roll 上一步估计。

    高速转动（|gyro_rate| ≥ rate_threshold）时运动加速度污染重力方向，
    完全信任陀螺积分；否则按 trust 融合陀螺预测与加速度计观测。
    单标量冒烟调用返回 0.0。
    """
    if gyro_rate is None and _is_scalar(accel):
        return 0.0

    a = np.asarray(accel, dtype=float).reshape(-1)
    rate = 0.0 if gyro_rate is None else float(np.asarray(gyro_rate).reshape(-1)[0])
    gravity_roll = math.atan2(float(a[1]), float(a[2]))
    predicted = float(roll) + rate * float(dt)

    alpha = 1.0 if abs(rate) >= float(rate_threshold) else float(trust)
    return alpha * predicted + (1.0 - alpha) * gravity_roll


def fov_compensate(angles, focal=None, frame_size=None) -> float:
    """计算旋转补偿后维持全覆盖所需的统一放大比例（≥1）。

    angles: 标量或 (3,) 旋转角（弧度），取最大绝对值 θ。
    仅给角度时按平面滚转保守估计 s = 1/cos θ（θ 需 < π/2）；
    同时给焦距 focal 与画面 frame_size 时，按角点半视场角
    φ=atan(c/f) 估计 s = tan(φ+θ)/tan φ，保证角点旋转后仍在画幅内。
    """
    theta = float(np.max(np.abs(np.asarray(angles, dtype=float).reshape(-1))))

    if focal is not None and frame_size is not None:
        width, height = float(frame_size[0]), float(frame_size[1])
        corner = math.hypot(width, height) / 2.0
        phi = math.atan2(corner, float(focal))
        ratio = math.tan(phi + theta) / math.tan(phi)
        return max(1.0, ratio)

    if theta >= math.pi / 2.0:
        raise ValueError("rotation angle must be smaller than pi/2 for FOV compensation")
    return max(1.0, 1.0 / math.cos(theta))


def batch(jobs, *, worker=None) -> list:
    """确定性批量驱动。

    jobs: 任务字典列表，每个含 id 与素材字段；按 id 字符串排序后处理，
    任务间无共享可变状态（缓存按 id 隔离），单任务异常被捕获并记录，
    不影响其余任务。worker 可注入自定义处理函数，默认走内置迷你流水线
    （校验 -> 降级/对齐 -> 确定性求解 -> 安全区裁剪）。
    单标量冒烟调用构造一个空素材任务并返回其报告。
    """
    if _is_scalar(jobs):
        jobs = [{"id": str(jobs)}]

    report = []
    for job in sorted(jobs, key=lambda j: str(j.get("id"))):
        try:
            result = _default_worker(job) if worker is None else worker(job)
            report.append({"id": job.get("id"), "ok": True, "result": result})
        except Exception as exc:  # 单项失败隔离
            report.append({
                "id": job.get("id"),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return report


def _default_worker(job: dict) -> dict:
    """内置单任务迷你流水线，供 batch 调用。"""
    gyro = job.get("gyro")
    frames = job.get("frames")
    frame_size = job.get("frame_size")
    margin = float(job.get("safe_margin", 0.0))
    radius = int(job.get("radius", 2))

    errors = validation(gyro, frames)
    if errors or gyro is None or frames is None:
        mode = fallback(gyro, frames)
    else:
        mode = {"mode": "gyro", "reason": "ok", "path": None}

    if mode["mode"] == "identity":
        path = mode["path"]
    else:
        aligned = gyro_timestamp(gyro, frames)
        # 以角速度累积位移作为原始路径，再走唯一确定性求解入口
        raw = np.cumsum(aligned, axis=0)[:, :2]
        path = solve_path(raw, radius)

    rect = None
    if frame_size is not None and len(path) > 0:
        width, height = float(frame_size[0]), float(frame_size[1])
        w = width * 0.9
        h = height * 0.9
        x = float(path[-1][0]) if path.ndim == 2 else float(path[-1])
        rect = crop_window((x, 0.0, w, h), frame_size, margin)

    return {
        "cache_key": f"{job.get('id')}:{job.get('lens_id', 'default')}",
        "errors": errors,
        "mode": mode["mode"],
        "path": np.asarray(path),
        "crop": rect,
    }
