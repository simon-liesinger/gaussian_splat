"""Verify that camera gradients are correct by comparing autodiff with finite differences.

This test creates a small synthetic scene, renders it, and checks that the
gradients w.r.t. camera quaternion and translation match numerical estimates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch
import torch.nn.functional as F

from camera_model import (
    quaternion_to_matrix,
    quaternion_multiply,
    transform_to_camera,
    matrix_to_quaternion,
)


def test_quaternion_to_matrix_gradient():
    """Test that quaternion_to_matrix is differentiable."""
    q = torch.tensor([1.0, 0.1, 0.2, 0.3], requires_grad=True)
    R = quaternion_to_matrix(q)
    loss = R.sum()
    loss.backward()

    assert q.grad is not None, "No gradient for quaternion"
    assert q.grad.shape == (4,), f"Wrong grad shape: {q.grad.shape}"
    print("PASS: quaternion_to_matrix gradient exists and has correct shape")


def test_transform_gradient_flows_to_camera():
    """Test that gradients flow from transformed positions back to camera params."""
    torch.manual_seed(42)

    means = torch.randn(100, 3)
    quats = F.normalize(torch.randn(100, 4), dim=-1)
    scales = torch.rand(100, 3) * 0.1

    cam_q = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    cam_t = torch.tensor([0.0, 0.0, 3.0], requires_grad=True)

    means_cam, quats_cam, scales_out = transform_to_camera(
        means, quats, scales, cam_q, cam_t
    )

    # Simulate a simple loss on transformed positions
    loss = means_cam.sum() + quats_cam.sum()
    loss.backward()

    assert cam_q.grad is not None, "No gradient for camera quaternion"
    assert cam_t.grad is not None, "No gradient for camera translation"
    assert not torch.all(cam_q.grad == 0), "Camera quaternion gradient is zero"
    assert not torch.all(cam_t.grad == 0), "Camera translation gradient is zero"

    print("PASS: gradients flow to camera quaternion and translation")


def test_camera_translation_gradient_numerical():
    """Compare autodiff translation gradient with finite differences."""
    torch.manual_seed(42)
    eps = 1e-7
    dtype = torch.float64  # float64 needed for accurate finite differences

    means = torch.randn(50, 3, dtype=dtype)
    quats = F.normalize(torch.randn(50, 4, dtype=dtype), dim=-1)
    scales = torch.rand(50, 3, dtype=dtype) * 0.1
    cam_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)

    def compute_loss(t):
        means_cam, _, _ = transform_to_camera(means, quats, scales, cam_q, t)
        return (means_cam ** 2).sum()

    cam_t = torch.tensor([0.0, 0.0, 3.0], dtype=dtype, requires_grad=True)
    loss = compute_loss(cam_t)
    loss.backward()
    autodiff_grad = cam_t.grad.clone()

    # Finite differences
    t_base = cam_t.data.clone()
    numerical_grad = torch.zeros(3, dtype=dtype)
    for i in range(3):
        t_plus = t_base.clone()
        t_plus[i] += eps
        t_minus = t_base.clone()
        t_minus[i] -= eps
        numerical_grad[i] = (compute_loss(t_plus).item() - compute_loss(t_minus).item()) / (2 * eps)

    rel_error = (autodiff_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)
    max_rel_error = rel_error.max().item()

    print(f"Translation gradient - max relative error: {max_rel_error:.6e}")
    assert max_rel_error < 1e-3, f"Translation gradient error too large: {max_rel_error}"
    print("PASS: translation gradient matches finite differences")


def test_camera_rotation_gradient_numerical():
    """Compare autodiff rotation gradient with finite differences."""
    torch.manual_seed(42)
    eps = 1e-7
    dtype = torch.float64

    means = torch.randn(50, 3, dtype=dtype)
    quats = F.normalize(torch.randn(50, 4, dtype=dtype), dim=-1)
    scales = torch.rand(50, 3, dtype=dtype) * 0.1
    cam_t = torch.tensor([0.0, 0.0, 3.0], dtype=dtype)

    def compute_loss(q):
        means_cam, quats_cam, _ = transform_to_camera(means, quats, scales, q, cam_t)
        return (means_cam ** 2).sum() + (quats_cam ** 2).sum()

    cam_q = torch.tensor([1.0, 0.1, -0.05, 0.2], dtype=dtype, requires_grad=True)
    loss = compute_loss(cam_q)
    loss.backward()
    autodiff_grad = cam_q.grad.clone()

    # Finite differences
    numerical_grad = torch.zeros(4, dtype=dtype)
    for i in range(4):
        q_plus = cam_q.data.clone()
        q_plus[i] += eps
        q_minus = cam_q.data.clone()
        q_minus[i] -= eps
        numerical_grad[i] = (compute_loss(q_plus).item() - compute_loss(q_minus).item()) / (2 * eps)

    rel_error = (autodiff_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)
    max_rel_error = rel_error.max().item()

    print(f"Rotation gradient - max relative error: {max_rel_error:.6e}")
    assert max_rel_error < 1e-3, f"Rotation gradient error too large: {max_rel_error}"
    print("PASS: rotation gradient matches finite differences")


def test_rotation_matrix_roundtrip():
    """Test quaternion -> matrix -> quaternion roundtrip."""
    q_orig = F.normalize(torch.tensor([0.5, 0.3, -0.1, 0.8]), dim=-1)
    R = quaternion_to_matrix(q_orig)
    q_back = matrix_to_quaternion(R)

    # Quaternions q and -q represent the same rotation
    if (q_orig - q_back).norm() > (q_orig + q_back).norm():
        q_back = -q_back

    error = (q_orig - q_back).abs().max().item()
    print(f"Quaternion roundtrip error: {error:.6e}")
    assert error < 1e-5, f"Roundtrip error too large: {error}"
    print("PASS: quaternion roundtrip")


if __name__ == "__main__":
    print("=" * 60)
    print("Camera Gradient Tests")
    print("=" * 60)

    test_quaternion_to_matrix_gradient()
    test_transform_gradient_flows_to_camera()
    test_camera_translation_gradient_numerical()
    test_camera_rotation_gradient_numerical()
    test_rotation_matrix_roundtrip()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
