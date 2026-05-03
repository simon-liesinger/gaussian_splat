"""Verify SE(3) camera model gradients by comparing autodiff with finite differences.

Tests that:
1. se3_exp is differentiable
2. CameraSet.viewmats() produces valid gradients in xi
3. Gradients match finite-difference numerical estimates (float64)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch
import torch.nn.functional as F

from camera_model import se3_exp, so3_exp, skew_symmetric, CameraSet, look_at, pose_to_matrix


def test_se3_exp_at_zero():
    """se3_exp(0) should be identity."""
    xi = torch.zeros(1, 6)
    T = se3_exp(xi)
    eye = torch.eye(4).unsqueeze(0)
    error = (T - eye).abs().max().item()
    assert error < 1e-6, f"se3_exp(0) != I, error={error}"
    print("PASS: se3_exp(0) = identity")


def test_se3_exp_differentiable():
    """se3_exp should produce gradients."""
    xi = torch.randn(5, 6, requires_grad=True)
    T = se3_exp(xi)
    loss = T.sum()
    loss.backward()
    assert xi.grad is not None, "No gradient"
    assert not torch.all(xi.grad == 0), "Zero gradient"
    print("PASS: se3_exp is differentiable")


def test_so3_exp_produces_rotation():
    """so3_exp should produce valid rotation matrices (R^T R = I, det = 1)."""
    omega = torch.randn(10, 3)
    R = so3_exp(omega)

    # R^T R should be identity
    RtR = R.transpose(-1, -2) @ R
    eye = torch.eye(3).unsqueeze(0).expand_as(RtR)
    error = (RtR - eye).abs().max().item()
    assert error < 1e-5, f"R^T R != I, error={error}"

    # det should be 1
    det = torch.det(R)
    det_error = (det - 1.0).abs().max().item()
    assert det_error < 1e-5, f"det(R) != 1, error={det_error}"
    print("PASS: so3_exp produces valid rotations")


def test_viewmats_gradient():
    """CameraSet.viewmats() should produce gradients in xi."""
    init_poses = torch.eye(4).unsqueeze(0).expand(3, -1, -1).clone()
    cameras = CameraSet(init_poses, device=torch.device("cpu"))

    viewmats = cameras.viewmats()
    loss = viewmats.sum()
    loss.backward()

    assert cameras.xi.grad is not None, "No gradient for xi"
    print("PASS: viewmats gradient flows to xi")


def test_xi_translation_gradient_numerical():
    """Compare autodiff gradient for translation part of xi with finite differences."""
    dtype = torch.float64
    eps = 1e-7

    init_poses = torch.eye(4, dtype=dtype).unsqueeze(0).expand(2, -1, -1).clone()
    # Set some non-trivial initial pose
    init_poses[0, :3, 3] = torch.tensor([0.0, 0.0, 3.0], dtype=dtype)
    init_poses[1, :3, 3] = torch.tensor([1.0, 0.0, 3.0], dtype=dtype)

    cameras = CameraSet(init_poses, device=torch.device("cpu"))
    cameras.xi.data = cameras.xi.data.to(dtype)
    cameras.R0 = cameras.R0.to(dtype)
    cameras.t0 = cameras.t0.to(dtype)

    def compute_loss(xi_val):
        cameras.xi.data.copy_(xi_val)
        vm = cameras.viewmats()
        return (vm ** 2).sum()

    xi_base = torch.randn(2, 6, dtype=dtype) * 0.1
    cameras.xi.data.copy_(xi_base)
    cameras.xi.requires_grad_(True)

    loss = compute_loss(xi_base)
    loss.backward()
    autodiff_grad = cameras.xi.grad.clone()

    # Finite differences
    numerical_grad = torch.zeros_like(xi_base)
    for i in range(2):
        for j in range(6):
            xi_plus = xi_base.clone()
            xi_plus[i, j] += eps
            xi_minus = xi_base.clone()
            xi_minus[i, j] -= eps

            cameras.xi.requires_grad_(False)
            l_plus = compute_loss(xi_plus).item()
            l_minus = compute_loss(xi_minus).item()
            numerical_grad[i, j] = (l_plus - l_minus) / (2 * eps)

    rel_error = (autodiff_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-15)
    max_rel_error = rel_error.max().item()

    print(f"xi gradient - max relative error: {max_rel_error:.6e}")
    assert max_rel_error < 1e-4, f"xi gradient error too large: {max_rel_error}"
    print("PASS: xi gradient matches finite differences")


def test_reanchor():
    """Reanchoring should preserve the viewmat but zero xi."""
    init_poses = torch.eye(4).unsqueeze(0).expand(3, -1, -1).clone()
    cameras = CameraSet(init_poses, device=torch.device("cpu"))

    # Set some non-zero xi
    with torch.no_grad():
        cameras.xi.data = torch.randn(3, 6) * 0.2

    viewmats_before = cameras.viewmats().detach().clone()
    cameras.reanchor()
    viewmats_after = cameras.viewmats().detach()

    # xi should be zero
    assert cameras.xi.data.abs().max() < 1e-8, "xi not zeroed after reanchor"

    # viewmats should be unchanged
    error = (viewmats_before - viewmats_after).abs().max().item()
    assert error < 1e-5, f"viewmats changed after reanchor, error={error}"
    print("PASS: reanchor preserves viewmats and zeros xi")


def test_look_at():
    """look_at should produce valid world-to-camera matrix."""
    eye = torch.tensor([0.0, 0.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])

    T = look_at(eye, target, up)

    # Should be a valid rigid transform
    R = T[:3, :3]
    det = torch.det(R)
    assert abs(det.item() - 1.0) < 1e-5, f"det(R) != 1: {det.item()}"

    # Camera at [0,0,3] looking at origin: origin should map to [0,0,-3] in camera space
    origin_cam = (R @ torch.tensor([0.0, 0.0, 0.0]) + T[:3, 3])
    # Z component should be negative (in front of camera looking down -Z)
    assert origin_cam[2] < 0, f"Origin not in front of camera: {origin_cam}"

    print("PASS: look_at produces valid camera matrix")


if __name__ == "__main__":
    print("=" * 60)
    print("SE(3) Camera Model Tests")
    print("=" * 60)

    test_se3_exp_at_zero()
    test_se3_exp_differentiable()
    test_so3_exp_produces_rotation()
    test_viewmats_gradient()
    test_xi_translation_gradient_numerical()
    test_reanchor()
    test_look_at()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
