"""Camera model with differentiable world-to-camera transform.

The key trick: instead of modifying the rasterizer to differentiate w.r.t.
camera parameters, we transform all splats into camera space before rasterizing.
The rasterizer's existing position/covariance gradients then flow back through
the transform into camera quaternion and translation parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    All operations are differentiable w.r.t. q.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def transform_to_camera(
    means: torch.Tensor,       # [N, 3] world positions
    quats: torch.Tensor,       # [N, 4] splat quaternions (w,x,y,z)
    scales: torch.Tensor,      # [N, 3] splat scales
    camera_q: torch.Tensor,    # [4] camera rotation quaternion
    camera_t: torch.Tensor,    # [3] camera translation
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform all splats into camera space.

    Moving splats by the inverse camera transform is equivalent to moving
    the camera. The rasterizer's position gradients then give us camera
    gradients for free via autograd.

    Returns: (means_cam, quats_cam, scales) -- scales unchanged.
    """
    R = quaternion_to_matrix(camera_q)  # [3, 3]

    # Transform positions: means_cam = R @ means^T + t
    means_cam = means @ R.T + camera_t.unsqueeze(0)  # [N, 3]

    # Rotate splat orientations into camera frame
    # q_cam = camera_q * q_splat  (quaternion multiplication)
    camera_q_expanded = camera_q.unsqueeze(0).expand(quats.shape[0], -1)
    quats_cam = quaternion_multiply(camera_q_expanded, quats)

    return means_cam, quats_cam, scales


def build_viewmat_identity(device: torch.device) -> torch.Tensor:
    """Identity 4x4 view matrix (camera at origin looking down -Z)."""
    return torch.eye(4, device=device, dtype=torch.float32)


class CameraSet(nn.Module):
    """Set of optimizable camera poses."""

    def __init__(self, num_cameras: int, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.num_cameras = num_cameras

        # Initialize camera quaternions and translations
        quats, trans = self._init_cameras(num_cameras)
        self.quats = nn.Parameter(quats.to(device))   # [N, 4]
        self.trans = nn.Parameter(trans.to(device))    # [N, 3]

    def _init_cameras(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize cameras distributed on a hemisphere looking at origin."""
        quats = []
        trans = []

        for i in range(n):
            # Distribute on hemisphere using golden spiral
            theta = math.acos(1 - (i + 0.5) / n)  # polar angle [0, pi/2]
            phi = math.pi * (1 + 5**0.5) * i       # azimuthal angle

            # Camera position on sphere of radius 3
            radius = 3.0
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            pos = torch.tensor([x, y, z])

            # Look at origin: compute rotation that aligns -Z with (origin - pos)
            forward = -pos / pos.norm()  # direction from camera to origin
            up = torch.tensor([0.0, 1.0, 0.0])

            # Handle degenerate case where forward is parallel to up
            if abs(forward[1]) > 0.99:
                up = torch.tensor([1.0, 0.0, 0.0])

            right = torch.cross(up, forward)
            right = right / right.norm()
            up = torch.cross(forward, right)

            # Build rotation matrix (camera convention: right=X, up=Y, forward=-Z)
            R = torch.stack([right, up, -forward], dim=0)  # [3, 3]

            # Convert to quaternion
            q = matrix_to_quaternion(R)
            quats.append(q)

            # Translation = R @ (-pos) -- world-to-camera
            t = R @ (-pos)
            trans.append(t)

        # Add small random perturbation
        quats = torch.stack(quats) + torch.randn(n, 4) * 0.02
        trans = torch.stack(trans) + torch.randn(n, 3) * 0.1

        return quats, trans

    def get_camera(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (quaternion, translation) for camera idx."""
        return self.quats[idx], self.trans[idx]


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / (trace + 1.0).sqrt()
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt()
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt()
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt()
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return torch.tensor([w, x, y, z])
