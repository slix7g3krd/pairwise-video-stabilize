"""手持拍摄视频防抖的辅助计算（第 61 批）。

每个函数对应一个防抖流水线中的缺陷修复点：时间戳对齐、滚动快门、
动态裁剪、地平线锁定、延迟补偿、VFR 平滑窗口、畸变顺序、切点重置、
高位深、多相机偏置、安全区、预览/导出一致性、缓存隔离。
"""

def timestamp_align(gyro_t, frame_t):
    """陀螺仪采样与帧时间戳对齐。

    返回两者之间的时间偏差（秒，取绝对值），用于把角速度序列
    重采样到帧时间轴上，避免"果冻"式的时域错位。
    """
    return abs(frame_t - gyro_t)

def rolling_shutter(line_time, exposure):
    """滚动快门校正量。

    每行读出时间为 line_time、曝光覆盖 exposure 行时，
    该帧的读出方向偏移为两者乘积。
    """
    return line_time * exposure

def dynamic_crop(jitter, base):
    """按抖动强度动态计算裁剪比例。

    jitter 为归一化抖动量，base 为基础缩放；抖动越大裁剪越紧，
    保证有足够余量做 warp。
    """
    return base / (1.0 + jitter)

def horizon_lock(accel, gyro, threshold=2.0):
    """地平线锁定。

    快速平移时加速度计会把运动加速度误判为重力方向，因此旋转速率
    超过阈值时以陀螺仪为准，否则采用加速度计给出的重力估计。
    """
    if abs(gyro) >= threshold:
        return gyro
    return accel

def latency_comp(sync_time, latency):
    """防抖延迟补偿。

    从同步时间中扣除渲染/传输延迟，使画面与声音对齐。
    """
    return sync_time - latency

def vfr_window(frames, rate):
    """VFR 素材的平滑窗口。

    按时间（帧数 / 帧率）而非固定帧数计算窗口长度，
    使可变帧率下窗口覆盖一致的时长。
    """
    return frames / rate

def distortion_order(distortion, stabilize):
    """畸变校正与防抖的先后顺序。

    返回 True 表示畸变校正应先于防抖执行；反之则顺序错误，
    warp 会把边缘拉不直。
    """
    return distortion < stabilize

def cut_reset(cut, trajectory):
    """场景切换点的轨迹重置。

    检测到切点后丢弃上一段运动轨迹，从零开始，避免转场漂移。
    """
    if cut:
        return 0
    return trajectory

def bit_depth(buffer_bits, source_bits):
    """高位深素材的缓冲位深。

    防抖必须在不低于素材位深的缓冲中进行，返回缓冲实际可用的位深。
    """
    return min(buffer_bits, source_bits)

def camera_bias(bias, camera):
    """多相机陀螺仪偏置的隔离键。

    偏置按相机分别估计与缓存，切换机位时重新标定。
    """
    return f"{camera}:{bias}"

def safe_area(crop, margin):
    """防抖后重新计算的安全区。

    在裁剪后的画面内扣除字幕等安全边距，避免字幕被裁。
    """
    return crop - margin

def preview_export_delta(preview, export):
    """预览与导出 warp 的一致性偏差。

    两者应使用同一套 warp 计算；返回其绝对差值用于校验一致性。
    """
    return abs(export - preview)

def cache_key(video_id, lens_id):
    """防抖缓存的隔离键。

    轨迹缓存需同时绑定视频与镜头，换镜头后不复用旧轨迹。
    """
    return f"{video_id}:{lens_id}"

def shake_energy(dx, dy):
    """抖动能量。

    用水平/垂直位移的绝对值和衡量抖动强度，用于动态裁剪与平滑强度。
    """
    return abs(dx) + abs(dy)

def smooth_window(radius):
    """平滑窗口长度。

    半径为 radius 的对称窗口包含中心帧，长度为 2*radius+1。
    """
    return 2 * radius + 1
