"""Main training script: COLMAP-free 3D Gaussian Splatting.

Jointly optimizes gaussian splat parameters and camera poses from scratch,
starting from random initialization. No SfM/COLMAP/pointcloud step needed.

Usage:
    python train.py --input video.mp4 --output output/
    python train.py --input images_dir/ --output output/
"""

import argparse
import os
import random
import time

import torch
import numpy as np
from tqdm import tqdm

from gaussian_model import GaussianModel
from camera_model import CameraSet, transform_to_camera, build_viewmat_identity
from densification import DensificationController
from loss import combined_loss
from utils import (
    extract_frames,
    load_images,
    frames_to_tensors,
    estimate_intrinsics,
    export_ply,
)

# Lazy import gsplat -- may not be installed during development
try:
    from gsplat.rendering import rasterization
except ImportError:
    rasterization = None


def make_optimizers(
    model: GaussianModel,
    cameras: CameraSet,
    position_lr: float = 1.6e-4,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Create separate optimizers for gaussians and cameras."""
    gaussian_params = [
        {"params": [model.means], "lr": position_lr, "name": "means"},
        {"params": [model.scales_raw], "lr": 5e-3, "name": "scales"},
        {"params": [model.quats_raw], "lr": 1e-3, "name": "quats"},
        {"params": [model.opacities_raw], "lr": 5e-2, "name": "opacities"},
        {"params": [model.sh_coeffs], "lr": 2.5e-3, "name": "sh"},
    ]

    camera_params = [
        {"params": [cameras.quats], "lr": 1e-4, "name": "cam_quats"},
        {"params": [cameras.trans], "lr": 1e-3, "name": "cam_trans"},
    ]

    g_opt = torch.optim.Adam(gaussian_params, eps=1e-15)
    c_opt = torch.optim.Adam(camera_params, eps=1e-15)

    return g_opt, c_opt


def rebuild_optimizers(
    model: GaussianModel,
    cameras: CameraSet,
    position_lr: float,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Rebuild optimizers after densification changes parameter tensors."""
    return make_optimizers(model, cameras, position_lr)


def position_lr_schedule(initial_lr: float, iteration: int, max_iters: int = 30000) -> float:
    """Exponential decay from initial_lr to initial_lr/100."""
    decay = (0.01) ** (iteration / max_iters)
    return initial_lr * decay


def render_frame(
    model: GaussianModel,
    cameras: CameraSet,
    cam_idx: int,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Render a single frame using gsplat.

    Returns [H, W, 3] float32 tensor.
    """
    params = model.get_activated()
    cam_q, cam_t = cameras.get_camera(cam_idx)

    # Transform splats into camera space
    means_cam, quats_cam, scales = transform_to_camera(
        params["means"], params["quats"], params["scales"],
        cam_q, cam_t,
    )

    # Build intrinsics matrix [3, 3]
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], device=means_cam.device, dtype=torch.float32)

    # Identity viewmat since we already transformed to camera space
    viewmat = build_viewmat_identity(means_cam.device)

    # gsplat rasterization
    renders, alphas, meta = rasterization(
        means=means_cam,
        quats=quats_cam,
        scales=scales,
        opacities=params["opacities"],
        colors=model.get_colors(),
        viewmats=viewmat.unsqueeze(0),   # [1, 4, 4]
        Ks=K.unsqueeze(0),               # [1, 3, 3]
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=100.0,
        render_mode="RGB",
    )

    return renders[0]  # [H, W, 3]


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input frames
    print("Loading input frames...")
    if os.path.isfile(args.input):
        frames = extract_frames(
            args.input,
            max_frames=args.max_frames,
            target_size=(args.width, args.height),
        )
    elif os.path.isdir(args.input):
        frames = load_images(
            args.input,
            target_size=(args.width, args.height),
        )
    else:
        raise ValueError(f"Input not found: {args.input}")

    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    target_images = frames_to_tensors(frames, device)

    height, width = frames[0].shape[:2]
    fx, fy, cx, cy = estimate_intrinsics(width, height)
    num_cameras = len(frames)

    # Initialize model
    print(f"Initializing {args.num_gaussians} gaussians and {num_cameras} cameras...")
    model = GaussianModel(
        num_gaussians=args.num_gaussians,
        sh_degree=0,
        device=device,
    )
    cameras = CameraSet(num_cameras, device=device)

    # Optimizers
    initial_position_lr = 1.6e-4
    g_opt, c_opt = make_optimizers(model, cameras, initial_position_lr)

    # Densification controller
    densifier = DensificationController(
        grad_threshold=args.densify_grad_threshold,
        scene_extent=3.0,
    )
    densifier.reset_accumulators(model.num_gaussians, device)

    # Output directory
    os.makedirs(args.output, exist_ok=True)

    # Training loop
    print(f"Training for {args.iterations} iterations...")
    start_time = time.time()

    for iteration in tqdm(range(args.iterations)):
        # Update position learning rate
        current_pos_lr = position_lr_schedule(initial_position_lr, iteration, args.iterations)
        for pg in g_opt.param_groups:
            if pg["name"] == "means":
                pg["lr"] = current_pos_lr

        # Sample random view
        cam_idx = random.randint(0, num_cameras - 1)
        target = target_images[cam_idx]

        # Zero gradients
        g_opt.zero_grad()
        c_opt.zero_grad()

        # Render
        rendered = render_frame(
            model, cameras, cam_idx,
            width, height, fx, fy, cx, cy,
        )

        # Loss
        loss = combined_loss(rendered, target, lambda_ssim=args.lambda_ssim)

        # Backward
        loss.backward()

        # Track gradients for densification
        if model.means.grad is not None:
            densifier.accumulate(model.means.grad)

        # Step optimizers
        g_opt.step()
        c_opt.step()

        # Densification
        if (args.densify_start <= iteration < args.densify_stop
                and iteration % args.densify_interval == 0):
            stats = densifier.densify(model)
            g_opt, c_opt = rebuild_optimizers(model, cameras, current_pos_lr)
            if iteration % 1000 == 0:
                tqdm.write(
                    f"[{iteration}] Densify: +{stats['cloned']} cloned, "
                    f"+{stats['split']*2} split, -{stats['pruned']} pruned "
                    f"= {stats['total']} total"
                )

        # Opacity reset
        if iteration > 0 and iteration % args.opacity_reset_interval == 0:
            model.reset_opacities()
            tqdm.write(f"[{iteration}] Reset opacities")

        # Logging
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            its = (iteration + 1) / elapsed if elapsed > 0 else 0
            tqdm.write(
                f"[{iteration}] loss={loss.item():.4f} "
                f"gaussians={model.num_gaussians} "
                f"it/s={its:.1f}"
            )

        # Save intermediate render
        if iteration % args.save_interval == 0 and iteration > 0:
            save_render(rendered, os.path.join(args.output, f"render_{iteration:06d}.png"))

    # Final export
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f}s ({args.iterations/elapsed:.1f} it/s)")

    params = model.get_activated()
    ply_path = os.path.join(args.output, "model.ply")
    export_ply(
        ply_path,
        params["means"],
        params["scales"],
        params["quats"],
        params["opacities"],
        model.sh_coeffs,
    )
    print(f"Exported {model.num_gaussians} gaussians to {ply_path}")

    # Save camera poses
    cam_path = os.path.join(args.output, "cameras.pt")
    torch.save({
        "quats": cameras.quats.detach().cpu(),
        "trans": cameras.trans.detach().cpu(),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "width": width,
        "height": height,
    }, cam_path)
    print(f"Saved camera poses to {cam_path}")


def save_render(image: torch.Tensor, path: str):
    """Save [H, W, 3] float tensor as PNG."""
    import cv2
    img = (image.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description="COLMAP-free 3D Gaussian Splatting")
    parser.add_argument("--input", required=True, help="Video file or image directory")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--width", type=int, default=400, help="Training resolution width")
    parser.add_argument("--height", type=int, default=400, help="Training resolution height")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to extract from video")
    parser.add_argument("--num-gaussians", type=int, default=10000, help="Initial gaussian count")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument("--lambda-ssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--densify-start", type=int, default=500)
    parser.add_argument("--densify-stop", type=int, default=15000)
    parser.add_argument("--densify-interval", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    parser.add_argument("--opacity-reset-interval", type=int, default=3000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    args = parser.parse_args()

    if rasterization is None:
        print("ERROR: gsplat is not installed. Install with: pip install gsplat")
        print("Requires CUDA GPU. See https://github.com/nerfstudio-project/gsplat")
        return

    train(args)


if __name__ == "__main__":
    main()
