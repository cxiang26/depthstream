"""Tests for flow utilities and affine re-anchoring (CPU-only, fast)."""
import numpy as np
import pytest

from depthstream.flow import align_affine, backward_warp, upsample_flow


def test_backward_warp_shifts_content():
    img = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    flow = np.zeros((20, 30, 2), np.float32)
    flow[..., 0] = 2.0                      # sample 2 px to the right
    out = backward_warp(img, flow)
    assert np.allclose(out[:, :-2], img[:, 2:], atol=1e-3)


def test_upsample_flow_scales_magnitudes():
    fl = np.ones((10, 10, 2), np.float32)
    up = upsample_flow(fl, up=4.0, size=(40, 40))
    assert up.shape == (40, 40, 2)
    assert np.allclose(up[..., 0], 4.0, atol=1e-3)


def test_align_affine_recovers_transform():
    rng = np.random.default_rng(0)
    a = (rng.random((100, 100)) * 5 + 1).astype(np.float64)
    b = 2.0 * a + 1.0
    alpha, beta = align_affine(a, b, np.ones_like(a, dtype=bool))
    assert abs(alpha - 2.0) < 1e-4
    assert abs(beta - 1.0) < 1e-4


def test_align_affine_constant_field_falls_back_to_identity():
    a = np.ones((50, 50), np.float32)
    b = np.full((50, 50), 3.0, np.float32)
    alpha, beta = align_affine(a, b, np.ones_like(a, dtype=bool))
    assert alpha == 1.0 and beta == 0.0          # documented safe fallback


def test_align_affine_tiny_mask_falls_back_to_identity():
    a = np.ones((50, 50), np.float32)
    b = np.full((50, 50), 3.0, np.float32)
    mask = np.zeros_like(a, dtype=bool)
    mask[:3, :3] = True                          # < 100 valid pixels
    alpha, beta = align_affine(a, b, mask)
    assert alpha == 1.0 and beta == 0.0
