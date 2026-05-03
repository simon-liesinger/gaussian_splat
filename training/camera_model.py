"""Camera model with SE(3) Lie-algebra parameterization.

Each camera stores a fixed reference pose (R0, t0) plus a learnable se(3)
tangent vector xi in R^6. The world-to-camera matrix is:

    viewmat = se3_exp(xi) @ [R0 | t0; 0 0 0 1]

This gives well-conditioned updates for large pose changes (cold start)
without quaternion normalization drift. Periodically call reanchor() to
fold xi back into (R0, t0) and reset xi to zero.

gsplat's rasterization() accepts viewmats as a differentiable [C, 4, 4]
tensor, so gradients flow into xi automatically — no wrapper needed.
"""

import torch
import torch.nn as nn
import math


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Convert [N, 3] vectors to [N, 3, 3] skew-symmetric matrices."""
    N = v.shape[0]
    zero = torch.zeros(N, device=v.device, dtype=v.dtype)
    return torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=-1).reshape(N, 3, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3). Rodrigues' formula.

    Args:
        omega: [N, 3] rotation vectors (axis * angle)

    Returns:
        [N, 3, 3] rotation matrices
    """
    theta = omega.norm(dim=-1, keepdim=True).unsqueeze(-1)  # [N, 1, 1]
    theta_safe = theta.clamp(min=1e-8)

    K = skew_symmetric(omega)  # [N, 3, 3]
    K2 = K @ K

    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    R = eye + (torch.sin(theta_safe) / theta_safe) * K + \
        ((1 - torch.cos(theta_safe)) / (theta_safe ** 2)) * K2

    return R


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map from se(3) to SE(3).

    Args:
        xi: [N, 6] tangent vectors. First 3 = rotation (omega), last 3 = translation (v).

    Returns:
        [N, 4, 4] transformation matrices
    """
    omega = xi[:, :3]  # [N, 3]
    v = xi[:, 3:]      # [N, 3]

    theta = omega.norm(dim=-1, keepdim=True).unsqueeze(-1)  # [N, 1, 1]
    theta_safe = theta.clamp(min=1e-8)

    K = skew_symmetric(omega)  # [N, 3, 3]
    K2 = K @ K

    # Rotation: Rodrigues
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0)
    sin_t = torch.sin(theta_safe) / theta_safe
    cos_t = (1 - torch.cos(theta_safe)) / (theta_safe ** 2)
    R = eye3 + sin_t * K + cos_t * K2

    # Translation: V matrix (left Jacobian of SO(3))
    V = eye3 + cos_t * K + \
        ((theta_safe - torch.sin(theta_safe)) / (theta_safe ** 3)) * K2

    t = (V @ v.unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # Build 4x4
    N = xi.shape[0]
    T = torch.zeros(N, 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T


def pose_to_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build [N, 4, 4] from [N, 3, 3] rotation and [N, 3] translation."""
    N = R.shape[0]
    T = torch.zeros(N, 4, 4, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute world-to-camera 4x4 matrix for a camera at eye looking at target.

    Args:
        eye: [3] camera position in world
        target: [3] point to look at
        up: [3] approximate up direction

    Returns:
        [4, 4] world-to-camera matrix
    """
    forward = target - eye
    forward = forward / forward.norm()

    right = torch.linalg.cross(forward, up)
    if right.norm() < 1e-6:
        up = torch.tensor([1.0, 0.0, 0.0], device=eye.device, dtype=eye.dtype)
        right = torch.linalg.cross(forward, up)
    right = right / right.norm()

    up = torch.linalg.cross(right, forward)

    # Camera convention: X=right, Y=up, Z=-forward (looking down -Z)
    R = torch.stack([right, up, -forward], dim=0)  # [3, 3]
    t = R @ (-eye)  # [3]

    T = torch.eye(4, device=eye.device, dtype=eye.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class CameraSet(nn.Module):
    """Set of optimizable camera poses parameterized in SE(3) tangent space."""

    def __init__(self, init_poses: torch.Tensor, device: torch.device = torch.device("cuda")):
        """
        Args:
            init_poses: [N, 4, 4] initial world-to-camera matrices
            device: torch device
        """
        super().__init__()
        init_poses = init_poses.to(device)
        self.num_cameras = init_poses.shape[0]

        self.register_buffer("R0", init_poses[:, :3, :3].contiguous())  # [N, 3, 3]
        self.register_buffer("t0", init_poses[:, :3, 3].contiguous())   # [N, 3]
        self.xi = nn.Parameter(torch.zeros(self.num_cameras, 6, device=device))

    def viewmats(self) -> torch.Tensor:
        """Return [N, 4, 4] world-to-camera matrices, differentiable in self.xi."""
        delta = se3_exp(self.xi)                     # [N, 4, 4]
        base = pose_to_matrix(self.R0, self.t0)      # [N, 4, 4]
        return delta @ base                          # left-multiply: refine in camera frame

    def reanchor(self):
        """Fold xi into (R0, t0) and reset xi to zero."""
        with torch.no_grad():
            new = self.viewmats()
            self.R0.copy_(new[:, :3, :3])
            self.t0.copy_(new[:, :3, 3])
            self.xi.zero_()

    @staticmethod
    def init_hemisphere(
        num_cameras: int,
        radius: float = 3.0,
        noise_rotation: float = 0.05,
        noise_translation: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ) -> "CameraSet":
        """Initialize cameras on a hemisphere looking at origin (for object-centric scenes)."""
        poses = []
        up = torch.tensor([0.0, 1.0, 0.0])
        target = torch.tensor([0.0, 0.0, 0.0])

        for i in range(num_cameras):
            # Golden spiral on hemisphere
            theta = math.acos(1 - (i + 0.5) / num_cameras)
            phi = math.pi * (1 + 5**0.5) * i

            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            eye = torch.tensor([x, y, z])

            pose = look_at(eye, target, up)
            poses.append(pose)

        poses = torch.stack(poses)

        # Add noise via small SE(3) perturbation
        cameras = CameraSet(poses, device=device)
        with torch.no_grad():
            cameras.xi[:, :3].normal_(0, noise_rotation)
            cameras.xi[:, 3:].normal_(0, noise_translation)
            cameras.reanchor()

        return cameras

    @staticmethod
    def init_sequential_arc(
        num_cameras: int,
        radius: float = 3.0,
        arc_degrees: float = 120.0,
        noise_rotation: float = 0.03,
        noise_translation: float = 0.05,
        device: torch.device = torch.device("cuda"),
    ) -> "CameraSet":
        """Initialize cameras along an arc (for video sequences)."""
        poses = []
        up = torch.tensor([0.0, 1.0, 0.0])
        target = torch.tensor([0.0, 0.0, 0.0])
        arc_rad = math.radians(arc_degrees)

        for i in range(num_cameras):
            frac = i / max(num_cameras - 1, 1)
            phi = -arc_rad / 2 + arc_rad * frac
            theta = math.pi / 4  # 45 degrees elevation

            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.sin(phi)
            eye = torch.tensor([x, y, z])

            pose = look_at(eye, target, up)
            poses.append(pose)

        poses = torch.stack(poses)

        cameras = CameraSet(poses, device=device)
        with torch.no_grad():
            cameras.xi[:, :3].normal_(0, noise_rotation)
            cameras.xi[:, 3:].normal_(0, noise_translation)
            cameras.reanchor()

        return cameras
