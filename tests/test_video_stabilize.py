import pytest
from media import video_stabilize as mod

CASES = [
    ("timestamp", 'm.timestamp_align(1.0, 3.0) == approx(2.0)'),
    ("rolling", 'm.rolling_shutter(0.01, 2.0) == approx(0.02)'),
    ("crop", 'm.dynamic_crop(0.5, 2.0) == approx(1.3333, rel=1e-3)'),
    ("horizon", 'm.horizon_lock(2.0, 1.0) == approx(2.0)'),
    ("latency", 'm.latency_comp(10, 2) == 8'),
    ("vfr", 'm.vfr_window(30, 30.0) == approx(1.0)'),
    ("distortion", 'm.distortion_order(1, 2) is True'),
    ("cut", 'm.cut_reset(True, 5) == 0'),
    ("bits", 'm.bit_depth(8, 10) == 8'),
    ("camera", 'm.camera_bias(1, 2) == "2:1"'),
    ("safe", 'm.safe_area(10, 2) == 8'),
    ("preview", 'm.preview_export_delta(1.0, 3.0) == approx(2.0)'),
    ("cache", 'm.cache_key("v", "l") == "v:l"'),
    ("shake", 'm.shake_energy(1.0, 2.0) == approx(3.0)'),
    ("window", 'm.smooth_window(2) == 5')
]

@pytest.mark.parametrize("name,expr", CASES, ids=[c[0] for c in CASES])
def test_contract(name, expr):
    assert eval(expr, {"m": mod, "approx": pytest.approx})
