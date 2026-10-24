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
    """Alternating 'model': even calls are wide linear fields (2+u), odd
    calls narrow quadratic fields (1.2+0.5u^2). Consecutive outputs differ
    in spread and shape, so the anchor fit has a measurable effect and
    contrast collapse is observable."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        h, w = frame.shape[:2]
        u = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        self.calls += 1
        if self.calls % 2 == 1:                       # call 0, 2, 4 ... wide
            return (2.0 + 1.0 * u).astype(np.float32)
        return (1.2 + 0.5 * u * u).astype(np.float32)  # narrow, same sign


def test_flow_runs_on_every_frame_with_an_anchor():
    # Regression guard for the keyframe path: due keyframes must also
    # compute flow+warp (they need the propagated field as the alignment
    # reference), so 10 frames at kf3 give 9 flow calls, not 6.
    st = DepthStream(GradientModel(), keyframe=3, flow="farneback")
    outs = [st.process(make_frame()) for _ in range(10)]
    assert st.n_infer == 4                        # t = 0, 3, 6, 9
    assert st.n_flow == 9


def _spread(o):
    return np.percentile(o, 95) - np.percentile(o, 5)


def test_due_keyframes_are_reanchored():
    # Static scene. At t=3 the narrow fresh field must be spread-matched
    # onto the wide anchor (alpha = sd ratio): the output's spread matches
    # the anchor's and the raw fresh field is overwritten.
    st = DepthStream(GradientModel(), keyframe=3, flow="farneback")
    outs = [st.process(make_frame()) for _ in range(10)]
    assert abs(_spread(outs[3]) / _spread(outs[0]) - 1.0) < 0.05
    u = np.tile(np.linspace(0, 1, 320, dtype=np.float32), (240, 1))
    assert not np.allclose(outs[3], 1.2 + 0.5 * u * u, atol=1e-3)


def test_contrast_preserved_under_chaining():
    # Regression guard for the collapse bug: iterated fits of fresh against
    # the (noisy) warped anchor attenuated the scale and collapsed the
    # chain's contrast to 0.44x raw. Spread matching must keep the chained
    # field's spread within 10% of the wide model output's.
    st = DepthStream(GradientModel(), keyframe=1, flow="farneback")
    outs = [st.process(make_frame()) for _ in range(30)]
    wide_spread = _spread(2.0 + np.tile(
        np.linspace(0, 1, 320, dtype=np.float32), (240, 1)))
    for o in outs[-10:]:
        assert 0.9 <= _spread(o) / wide_spread <= 1.1


def test_ema_blends_at_due_keyframes():
    # EMA must be active on due keyframes (not only on adaptive
    # promotions): with lambda=0.6 the output differs from pure
    # re-anchoring wherever the aligned field misses the anchor (the
    # narrow quadratic shape cannot be affine-matched onto the anchor).
    st1 = DepthStream(GradientModel(), keyframe=3, flow="farneback", ema=1.0)
    st2 = DepthStream(GradientModel(), keyframe=3, flow="farneback", ema=0.6)
    o1 = [st1.process(make_frame()) for _ in range(6)]
    o2 = [st2.process(make_frame()) for _ in range(6)]
    assert st1.n_infer == st2.n_infer == 2
    assert np.isfinite(o2[3]).all()
    assert not np.allclose(o1[3], o2[3], atol=1e-4)
