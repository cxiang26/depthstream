"""Tests for the DepthStream scheduling logic (fake model, CPU-only)."""
import numpy as np

from depthstream import DepthStream


class FakeModel:
    """Constant-depth 'model' that counts invocations."""

    def __init__(self, value=1.0):
        self.calls = 0
        self.value = value

    def __call__(self, frame):
        self.calls += 1
        return np.full((224, 224), self.value, np.float32)


def make_frame(color=0):
    return np.full((240, 320, 3), color, np.uint8)


def test_fixed_cadence_calls_model_every_k_frames():
    model = FakeModel()
    st = DepthStream(model, keyframe=10, flow="farneback")
    for _ in range(30):
        st.process(make_frame())
    assert st.i == 30
    assert st.n_infer == 3                       # t=0, 10, 20
    assert 24 <= st.n_flow <= 30                 # propagated frames use flow


def test_keyframe_one_is_dense():
    model = FakeModel()
    st = DepthStream(model, keyframe=1, flow="farneback")
    for _ in range(15):
        st.process(make_frame())
    assert st.n_infer == 15


def test_stats_report_latency_and_counts():
    st = DepthStream(FakeModel(), keyframe=5, flow="farneback")
    for _ in range(10):
        st.process(make_frame())
    s = st.stats()
    assert s["frames"] == 10
    assert s["n_infer"] == 2
    assert "ms_per_frame" in s


def test_adaptive_scheduler_triggers_on_scene_change():
    model = FakeModel()
    st = DepthStream(model, keyframe=10, flow="farneback",
                     adaptive=True, min_key_gap=2)
    # first three black frames, then white frames: photometric residual
    # exceeds the threshold -> early re-anchor expected at t=2 (gap >= 2)
    for t in range(8):
        st.process(make_frame(0 if t < 3 else 255))
    assert st.n_adaptive >= 1
    assert st.n_infer == 2                        # t=0 keyframe + t=2 adaptive
