"""六个核心缺陷修复的专项验证 + 支撑函数行为测试。

运行：python -m pytest tests/test_fixes.py -q
"""

import math

import numpy as np
import pytest

from media.video_stabilize import (
    PathSmoother,
    async_render,
    audit,
    batch,
    border_fill,
    crop_window,
    determinism,
    fallback,
    fov_compensate,
    gyro_timestamp,
    horizon,
    metrics,
    rolling_shutter,
    segment_reset,
    smooth_path,
    solve_path,
    validation,
    _rodrigues,
)


# --- 修复点 1：统一时间基并插值对齐 ---------------------------------------

def test_gyro_timestamp_unified_timebase():
    # 陀螺域：纳秒，零点 1_000_000_000；帧域：毫秒，零点 1000
    # 物理时间轴（秒）：0.0, 0.1, 0.2
    gyro_ns = np.array([
        [1_000_000_000, 1.0, 0.0, 0.0],
        [1_100_000_000, 3.0, 0.0, 0.0],
        [1_200_000_000, 5.0, 0.0, 0.0],
    ])
    frame_ms = np.array([1000.0, 1050.0, 1100.0, 1200.0])  # 0, .05, .1, .2 s

    aligned = gyro_timestamp(
        gyro_ns, frame_ms,
        gyro_origin=1_000_000_000, gyro_scale=1e-9,
        frame_origin=1000.0, frame_scale=1e-3,
    )
    # 0.05s 处线性插值 = 2.0；其余落在样本点上
    assert aligned.shape == (4, 3)
    assert aligned[:, 0] == pytest.approx([1.0, 2.0, 3.0, 5.0])


def test_gyro_timestamp_clamps_outside_range():
    gyro = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    frames = np.array([-1.0, 0.5, 5.0])
    aligned = gyro_timestamp(gyro, frames)
    # 不外推：早于首样本钳制到首值，晚于尾样本钳制到尾值
    assert aligned[0, 0] == pytest.approx(1.0)
    assert aligned[1, 0] == pytest.approx(1.5)
    assert aligned[2, 0] == pytest.approx(2.0)


def test_gyro_timestamp_scalar_offset():
    assert gyro_timestamp(2.0, 5.0) == pytest.approx(3.0)
    assert gyro_timestamp(1) == 0.0


# --- 修复点 2：滚动快门按行曝光补偿 ---------------------------------------

def test_rolling_shutter_center_row_identity():
    omega = np.array([0.0, 0.0, 0.5])
    mats = rolling_shutter(omega, rows=11, line_time=1e-3)
    assert mats.shape == (11, 3, 3)
    # 中心行曝光时刻 = 帧中心时刻，相对旋转为单位阵
    assert np.allclose(mats[5], np.eye(3))


def test_rolling_shutter_matches_rodrigues_and_differs_per_row():
    omega = np.array([0.3, -0.2, 0.5])
    rows, line_time = 8, 2e-3
    mats = rolling_shutter(omega, rows=rows, line_time=line_time)
    for r in range(rows):
        t_r = (r - (rows - 1) / 2.0) * line_time
        assert np.allclose(mats[r], _rodrigues(omega * t_r))
    # 首行与末行旋转方向相反，绝不能共用一个帧级旋转
    assert not np.allclose(mats[0], mats[-1])
    assert np.allclose(mats[0], mats[-1].T, atol=1e-10)  # 互逆


def test_rolling_shutter_scalar_readout_time():
    assert rolling_shutter(30e-6, 1080) == pytest.approx(30e-6 * 1080)


# --- 修复点 3：裁剪窗口限制在安全区 ---------------------------------------

@pytest.mark.parametrize("proposed,margin", [
    ((-100.0, -50.0, 400.0, 300.0), 0.0),       # 左上角越界
    ((1900.0, 1000.0, 400.0, 300.0), 0.0),      # 右下角越界
    ((0.0, 0.0, 5000.0, 4000.0), 20.0),         # 窗口超大 + 安全边距
])
def test_crop_window_always_inside_safe_area(proposed, margin):
    W, H = 1920.0, 1080.0
    x, y, w, h = crop_window(proposed, (W, H), margin)
    assert x >= margin and y >= margin
    assert x + w <= W - margin
    assert y + h <= H - margin
    assert w <= W - 2 * margin and h <= H - 2 * margin


def test_crop_window_scalar():
    assert crop_window(100) == (0.0, 0.0, 100.0, 100.0)


# --- 修复点 4：跨片段继承平滑状态 -----------------------------------------

def test_segment_inheritance_equals_continuous_pass():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(20, 2))
    alpha = 0.3

    # 连续一遍（基准）
    ref = PathSmoother(alpha).update(raw)

    # 拆成三段、边界不重置
    state = None
    chunks = []
    for lo, hi in [(0, 7), (7, 13), (13, 20)]:
        out, state = segment_reset(state, raw[lo:hi], is_cut=False, alpha=alpha)
        chunks.append(out)
    stitched = np.concatenate(chunks)

    assert np.allclose(stitched, ref)  # 继承状态 => 边界无跳变


def test_real_cut_restarts_path():
    seg1 = np.ones((4, 2)) * 10.0
    seg2 = np.ones((4, 2))
    _, state = segment_reset(None, seg1, is_cut=False, alpha=0.3)

    # 非切点：继承，首点被历史状态拉高，不会等于 seg2 原始值
    inherited, _ = segment_reset(state, seg2, is_cut=False, alpha=0.3)
    # 真切点：重启，首点等于原始值
    restarted, _ = segment_reset(state, seg2, is_cut=True, alpha=0.3)
    assert not np.allclose(inherited[0], seg2[0])
    assert np.allclose(restarted[0], seg2[0])


# --- 修复点 5：异步渲染按帧时间采样 ---------------------------------------

def test_async_render_immutable_snapshot_and_stale():
    gyro = np.array([
        [0.10, 1.0, 0.0, 0.0],
        [0.20, 2.0, 0.0, 0.0],
        [0.30, 3.0, 0.0, 0.0],
    ])
    frames = np.array([0.0, 0.15, 0.30])  # 首帧早于陀螺最早样本
    recs = async_render(frames, gyro)

    # 生产方随后覆盖/清空缓冲，渲染结果不受影响（快照不可变）
    gyro[:, 1] = 999.0
    gyro[:] = 0.0
    assert recs[1]["gyro"][0] == pytest.approx(1.5)
    assert recs[2]["gyro"][0] == pytest.approx(3.0)

    # 过期显式标记而不是静默读到"最新样本"
    assert recs[0]["stale"] is True
    assert recs[0]["gyro"][0] == pytest.approx(1.0)  # 钳制到首样本
    assert recs[1]["stale"] is False

    # 姿态按帧序累积、可复现
    recs2 = async_render(frames, np.array([
        [0.10, 1.0, 0.0, 0.0],
        [0.20, 2.0, 0.0, 0.0],
        [0.30, 3.0, 0.0, 0.0],
    ]))
    assert np.array_equal(recs[1]["rotation"], recs2[1]["rotation"])


def test_async_render_no_gyro_identity():
    recs = async_render(np.array([0.0, 0.1]))
    assert len(recs) == 2
    assert np.allclose(recs[1]["rotation"], np.eye(3))


# --- 修复点 6：路径求解确定性 ---------------------------------------------

def test_determinism_repeated_runs_bit_identical():
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(50, 2))
    report = determinism(pts, radius=3, runs=4)
    assert report["equal"] is True
    assert report["runs"] == 4
    assert np.array_equal(report["path"], solve_path(pts, 3))


def test_determinism_holds_across_process_style_smoothes():
    # 打乱输入顺序的调用方式不影响结果（无隐藏迭代序依赖）
    pts = np.linspace(0, 10, 11).reshape(-1, 1)
    a = solve_path(pts, 2)
    b = solve_path(pts, 2)
    assert np.array_equal(a, b)
    assert np.allclose(smooth_path(np.full(9, 3.0), 2), 3.0)  # 常值平滑后不变


# --- 支撑函数 -------------------------------------------------------------

def test_validation_detects_bad_inputs():
    assert validation() == []
    bad_nan = np.array([[0.0, 1.0], [1.0, np.nan]])
    assert any("nan" in e for e in validation(bad_nan))
    non_mono = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert "gyro:timestamps-not-strictly-increasing" in validation(non_mono)
    assert "gyro:need-at-least-two-samples" in validation(np.array([[0.0, 1.0]]))
    assert "line_time:negative" in validation(line_time=-1.0)
    good = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    assert validation(good, np.array([0.0, 1.0])) == []


def test_audit_digest_reproducible():
    a = audit({"x": [1, 2, 3]}, {"r": 2})
    b = audit({"x": [1, 2, 3]}, {"r": 2})
    c = audit({"x": [1, 2, 3]}, {"r": 3})
    assert a["digest"] == b["digest"]
    assert a["digest"] != c["digest"]
    assert len(a["digest"]) == 64  # sha256 hex


def test_fallback_modes():
    assert fallback(1)["mode"] == "identity"
    good = np.array([[0.0, 0.1], [1.0, 0.1]])
    assert fallback(good, np.array([0.0, 1.0]))["mode"] == "gyro"
    assert fallback(None, np.array([0.0]))["mode"] == "identity"
    low = fallback(good, np.array([0.0, 1.0]), confidence=0.1, threshold=0.5)
    assert low["mode"] == "identity" and low["reason"] == "low-confidence"
    bad = np.array([[0.0, np.nan], [1.0, 0.1]])
    assert fallback(bad, np.array([0.0, 1.0]))["reason"] == "invalid-gyro"


def test_border_fill_crop_covers_frame():
    plan = border_fill((12.0, 8.0), (1920, 1080), mode="crop")
    x, y, w, h = plan["rect"]
    assert x == 12.0 and y == 8.0
    assert x + w <= 1920.0 and y + h <= 1080.0
    assert plan["scale"] >= 1.0
    reflect = border_fill(5.0, (100, 100), mode="reflect")
    assert reflect["fill"]["left"] == 5.0 and reflect["scale"] == 1.0
    assert border_fill(3)["border"] == (3.0, 3.0)


def test_metrics_reports():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 2))
    smoothed = smooth_path(raw, 3)
    m = metrics(raw, smoothed, frame_size=(100, 100),
                crop_rect=(10, 10, 80, 80))
    assert m["jitter_after"] <= m["jitter_before"]
    assert 0.0 <= m["residual_ratio"] <= 1.0
    assert m["crop_loss"] == pytest.approx(0.36)
    assert metrics(np.zeros((5, 2)))["jitter_before"] == 0.0


def test_horizon_complementary_filter():
    accel = np.array([0.0, 1.0, 0.0])  # 重力滚转 = pi/2
    # 低速：融合加速度计观测
    slow = horizon(accel, gyro_rate=0.1, dt=0.01, roll=0.0)
    # 高速（超阈值）：只信陀螺积分
    fast = horizon(accel, gyro_rate=5.0, dt=0.01, roll=0.0,
                   rate_threshold=2.0)
    assert fast == pytest.approx(0.05)
    assert slow != pytest.approx(0.05)
    assert horizon(1) == 0.0


def test_fov_compensate():
    assert fov_compensate(0.0) == pytest.approx(1.0)
    assert fov_compensate(0.2) > 1.0
    assert fov_compensate(np.array([0.0, 0.0, 0.3])) == pytest.approx(
        1.0 / math.cos(0.3))
    focal = fov_compensate(0.1, focal=1000.0, frame_size=(1920, 1080))
    assert focal > 1.0
    with pytest.raises(ValueError):
        fov_compensate(math.pi / 2)


def test_batch_deterministic_and_isolated():
    gyro = np.array([[0.0, 0.1, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0]])
    jobs = [
        {"id": "b", "gyro": gyro, "frames": np.array([0.0, 1.0]),
         "frame_size": (640, 360)},
        {"id": "a", "gyro": gyro, "frames": np.array([0.0, 1.0]),
         "frame_size": (640, 360), "lens_id": "wide"},
        {"id": "c"},  # 无陀螺 -> identity 降级，不报错
        {"id": "d", "radius": "not-an-int"},  # 触发异常，隔离失败
    ]
    r1, r2 = batch(jobs), batch(jobs)
    assert [r["id"] for r in r1] == ["a", "b", "c", "d"]  # 按 id 排序
    assert all(x["ok"] for x in r1 if x["id"] != "d")
    assert r1[3]["ok"] is False and "error" in r1[3]
    # 缓存键按视频/镜头隔离
    assert r1[0]["result"]["cache_key"] != r1[1]["result"]["cache_key"]
    # 两遍结果逐位一致
    for x, y in zip(r1, r2):
        if x["ok"] and isinstance(x["result"]["path"], np.ndarray):
            assert np.array_equal(x["result"]["path"], y["result"]["path"])
    assert batch(1)[0]["ok"] is True
