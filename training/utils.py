"""Utilities: video frame extraction, image loading, PLY export."""

import os
import struct
import numpy as np
import torch
import cv2
from pathlib import Path


def extract_frames(
    video_path: str,
    max_frames: int = 200,
    target_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Extract frames from video file.

    Args:
        video_path: path to video file
        max_frames: maximum number of frames to extract
        target_size: (width, height) to resize to, or None for original size

    Returns:
        list of [H, W, 3] uint8 numpy arrays (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")

    # Sample frames evenly
    step = max(1, total_frames // max_frames)
    indices = list(range(0, total_frames, step))[:max_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()
    return frames


def load_images(
    image_dir: str,
    target_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Load images from directory.

    Returns:
        list of [H, W, 3] uint8 numpy arrays (RGB)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in exts and not p.name.startswith("._")
    )

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        frames.append(img)

    return frames


def frames_to_tensors(
    frames: list[np.ndarray],
    device: torch.device = torch.device("cuda"),
) -> list[torch.Tensor]:
    """Convert list of uint8 numpy frames to float32 tensors in [0, 1]."""
    return [
        torch.from_numpy(f).float().to(device) / 255.0
        for f in frames
    ]


def estimate_intrinsics(
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Estimate camera intrinsics (fx, fy, cx, cy) assuming typical phone camera.

    Returns reasonable defaults when no calibration data is available.
    """
    focal = max(width, height)
    return focal, focal, width / 2.0, height / 2.0


def export_ply(
    path: str,
    means: torch.Tensor,       # [N, 3]
    scales: torch.Tensor,      # [N, 3]
    quats: torch.Tensor,       # [N, 4]
    opacities: torch.Tensor,   # [N]
    sh_coeffs: torch.Tensor,   # [N, K, 3]
):
    """Export gaussians to PLY format compatible with standard viewers."""
    means = means.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    quats = quats.detach().cpu().numpy()
    opacities = opacities.detach().cpu().numpy()
    sh_coeffs = sh_coeffs.detach().cpu().numpy()

    n = means.shape[0]
    num_sh = sh_coeffs.shape[1]

    # Build PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
"""

    # SH coefficients (interleaved as f_dc_0, f_dc_1, f_dc_2, f_rest_0, ...)
    for i in range(3):
        header += f"property float f_dc_{i}\n"
    for i in range((num_sh - 1) * 3):
        header += f"property float f_rest_{i}\n"

    header += "property float opacity\n"
    header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode())

        for i in range(n):
            # Position
            f.write(struct.pack("fff", *means[i]))
            # Normals (unused, set to 0)
            f.write(struct.pack("fff", 0.0, 0.0, 0.0))
            # SH DC
            f.write(struct.pack("fff", *sh_coeffs[i, 0]))
            # SH rest
            for j in range(1, num_sh):
                f.write(struct.pack("fff", *sh_coeffs[i, j]))
            # Opacity (logit space for standard viewers)
            f.write(struct.pack("f", float(np.log(opacities[i] / (1 - opacities[i] + 1e-8)))))
            # Scale (log space)
            f.write(struct.pack("fff", *np.log(scales[i])))
            # Rotation quaternion
            f.write(struct.pack("ffff", *quats[i]))
