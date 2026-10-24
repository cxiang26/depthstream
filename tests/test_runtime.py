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


class GradientModel:
    """Alternating-gradient 'model': call k returns base +/- slope*u.

    Consecutive outputs are not affine-aligned (slope flips), so a due
    keyframe's re-anchoring has a measurable effect.
    """

    def __init__(self, base=2.0, slope=1.0):
        self.calls = 0
        self.base = base
        self.slope = slope

    def __call__(self, frame):
        h, w = frame.shape[:2]
        u = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        s = self.slope if self.calls % 2 == 0 else -self.slope
        self.calls += 1
        return (self.base + s * u).astype(np.float32)


def test_flow_runs_on_every_frame_with_an_anchor():
    # Regression guard for the keyframe path: due keyframes must also
    # compute flow+warp (they need the propagated field as the alignment
    # reference), so 10 frames at kf3 give 9 flow calls, not 6.
    st = DepthStream(GradientModel(), keyframe=3, flow="farneback")
    outs = [st.process(make_frame()) for _ in range(10)]
    assert st.n_infer == 4                        # t = 0, 3, 6, 9
    assert st.n_flow == 9


def test_due_keyframes_are_reanchored():
    # Static scene, model slope flips each call. At t=3 the fresh field
    # (3-u) must be aligned onto the propagated anchor (2+u): with the
    # scale clipped to 0.25 the output is 2.625-0.25u, not the raw 3-u.
    st = DepthStream(GradientModel(), keyframe=3, flow="farneback")
    outs = [st.process(make_frame()) for _ in range(10)]
    u = np.tile(np.linspace(0, 1, 320, dtype=np.float32), (240, 1))
    expected = 2.625 - 0.25 * u
    assert np.allclose(outs[3], expected, atol=1e-3)
    assert not np.allclose(outs[3], 3.0 - u, atol=1e-3)


def test_ema_blends_at_due_keyframes():
    # EMA must be active on due keyframes (not only on adaptive
    # promotions): with lambda=0.6 the output differs from pure
    # re-anchoring wherever the aligned field misses the anchor.
    st1 = DepthStream(GradientModel(), keyframe=3, flow="farneback", ema=1.0)
    st2 = DepthStream(GradientModel(), keyframe=3, flow="farneback", ema=0.6)
    o1 = [st1.process(make_frame()) for _ in range(6)]
    o2 = [st2.process(make_frame()) for _ in range(6)]
    assert st1.n_infer == st2.n_infer == 2
    assert np.isfinite(o2[3]).all()
    assert not np.allclose(o1[3], o2[3], atol=1e-4)
