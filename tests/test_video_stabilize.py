import pytest
import media.video_stabilize as mod

CASES = [
    ("gyro_timestamp", (1,)),
    ("rolling_shutter", (1,)),
    ("crop_window", (1,)),
    ("segment_reset", (1,)),
    ("async_render", (1,)),
    ("determinism", (1,)),
    ("smooth_path", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("border_fill", (1,)),
    ("metrics", (1,)),
    ("horizon", (1,)),
    ("fov_compensate", (1,)),
    ("batch", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
