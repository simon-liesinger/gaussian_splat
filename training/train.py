"""Main training script: COLMAP-free 3D Gaussian Splatting.

Jointly optimizes gaussian splat parameters and camera poses from scratch,
starting from random initialization. No SfM/COLMAP/pointcloud step needed.

Camera gradients flow through gsplat's native viewmats differentiability.
Poses are parameterized as SE(3) tangent vectors for well-conditioned updates.

Usage:
    python train.py --input video.mp4 --output output/
    python train.py --input images_dir/ --output output/
    python train.py --input images_dir/ --output output/ --stages A   # warmup only
    python train.py --input images_dir/ --output output/ --stages A,B # warmup + joint
"""

import argparse
import os
import random
import time
import json
import math

import torch
import numpy as np
from tqdm import tqdm

from gaussian_model import GaussianModel
from camera_model import CameraSet
from densification import DensificationController
from loss import combined_loss
from utils import (
    extract_frames,
    load_images,
    frames_to_tensors,
    estimate_intrinsics,
    export_ply,
)

try:
    from gsplat.rendering import rasterization
except ImportError:
    rasterization = None


def make_gaussian_optimizer(model: GaussianModel, position_lr: float = 1.6e-4) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": [model.means], "lr": position_lr, "name": "means"},
        {"params": [model.scales_raw], "lr": 5e-3, "name": "scales"},
        {"params": [model.quats_raw], "lr": 1e-3, "name": "quats"},
        {"params": [model.opacities_raw], "lr": 5e-2, "name": "opacities"},
        {"params": [model.sh_coeffs], "lr": 2.5e-3, "name": "sh"},
    ], eps=1e-15)


def make_camera_optimizer(cameras: CameraSet, lr_rot: float = 1e-3, lr_trans: float = 1e-2) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": [cameras.xi], "lr": lr_rot, "name": "camera_xi"},
    ], eps=1e-15)


ALL_STAGES = {
    "A": {"name": "A_warmup", "res": 200, "iters": 3000,
         "cam_lr_rot": 1e-3, "cam_lr_trans": 1e-2,
         "gauss_lr_scale": 0.1, "densify": False},
    "B": {"name": "B_joint", "res": 400, "iters": 12000,
         "cam_lr_rot": 5e-4, "cam_lr_trans": 5e-3,
         "gauss_lr_scale": 1.0, "densify": True},
    "C": {"name": "C_detail", "res": None, "iters": 15000,
         "cam_lr_rot": 1e-5, "cam_lr_trans": 1e-4,
         "gauss_lr_scale": 1.0, "densify": True},
}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input frames at full resolution
    print("Loading input frames...")
    if os.path.isfile(args.input):
        frames_full = extract_frames(args.input, max_frames=args.max_frames)
    elif os.path.isdir(args.input):
        frames_full = load_images(args.input)
    else:
        raise ValueError(f"Input not found: {args.input}")

    num_cameras = len(frames_full)
    if num_cameras < 2:
        raise ValueError(f"Need at least 2 frames, got {num_cameras}. Check input path and file extensions.")
    h_full, w_full = frames_full[0].shape[:2]
    print(f"Loaded {num_cameras} frames at {w_full}x{h_full}")

    # Select stages
    stage_keys = [s.strip().upper() for s in args.stages.split(",")]
    stages = []
    for k in stage_keys:
        if k not in ALL_STAGES:
            raise ValueError(f"Unknown stage '{k}'. Valid: A, B, C")
        stage = dict(ALL_STAGES[k])
        if stage["res"] is None:
            stage["res"] = min(800, max(w_full, h_full))
        stages.append(stage)
    print(f"Running stages: {', '.join(s['name'] for s in stages)}")

    # Initialize model
    print(f"Initializing {args.num_gaussians} gaussians and {num_cameras} cameras...")
    try:
        model = GaussianModel.from_images(
            frames_full, num_gaussians=args.num_gaussians, device=device,
        )
        print(f"Initialized splat colors from image k-means")
    except Exception as e:
        print(f"Color init failed ({e}), using random init")
        model = GaussianModel(
            num_gaussians=args.num_gaussians,
            sh_degree=0,
            device=device,
        )

    cameras = CameraSet.init_sequential_arc(
        num_cameras=num_cameras,
        radius=3.0,
        arc_degrees=120.0,
        device=device,
    )

    densifier = DensificationController(
        grad_threshold=args.densify_grad_threshold,
        scene_extent=3.0,
    )

    os.makedirs(args.output, exist_ok=True)

    log = {"stages": [], "iterations": []}

    global_iter = 0
    start_time = time.time()

    for stage in stages:
        stage_name = stage["name"]
        res = stage["res"]
        stage_iters = stage["iters"]
        print(f"\n{'='*60}")
        print(f"Stage {stage_name}: {res}x{res}, {stage_iters} iterations")
        print(f"{'='*60}")

        import cv2
        target_images = []
        for f in frames_full:
            resized = cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(resized).float().to(device) / 255.0
            target_images.append(t)

        width, height = res, res
        fx, fy, cx, cy = estimate_intrinsics(width, height)

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], device=device, dtype=torch.float32)
        Ks = K.unsqueeze(0).expand(num_cameras, -1, -1)

        pos_lr = 1.6e-4 * stage["gauss_lr_scale"]
        g_opt = make_gaussian_optimizer(model, position_lr=pos_lr)
        c_opt = make_camera_optimizer(cameras, lr_rot=stage["cam_lr_rot"], lr_trans=stage["cam_lr_trans"])

        densifier.reset_accumulators(model.num_gaussians, device)

        for local_iter in tqdm(range(stage_iters), desc=stage_name):
            # Render multiple views per iteration for 3D consistency
            n_views = min(args.views_per_iter, num_cameras)
            cam_idxs = random.sample(range(num_cameras), n_views)

            g_opt.zero_grad()
            c_opt.zero_grad()

            viewmats = cameras.viewmats()

            params = model.get_activated()
            renders, alphas, meta = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=params["scales"],
                opacities=params["opacities"],
                colors=model.get_colors(),
                viewmats=viewmats[cam_idxs],
                Ks=Ks[cam_idxs],
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100.0,
                render_mode="RGB",
            )

            loss = sum(
                combined_loss(renders[i], target_images[cam_idxs[i]], lambda_ssim=args.lambda_ssim)
                for i in range(n_views)
            ) / n_views

            # For snapshots, keep the first rendered view
            rendered = renders[0]
            cam_idx = cam_idxs[0]

            loss.backward()

            if stage["densify"] and model.means.grad is not None:
                densifier.accumulate(model.means.grad)

            g_opt.step()
            c_opt.step()

            if local_iter > 0 and local_iter % 500 == 0:
                cameras.reanchor()
                c_opt = make_camera_optimizer(cameras, lr_rot=stage["cam_lr_rot"], lr_trans=stage["cam_lr_trans"])

            if (stage["densify"]
                    and local_iter >= 500
                    and local_iter % 100 == 0
                    and local_iter < stage_iters - 1000):
                stats = densifier.densify(model)
                g_opt = make_gaussian_optimizer(model, position_lr=pos_lr)
                if local_iter % 1000 == 0:
                    tqdm.write(
                        f"[{global_iter}] Densify: +{stats['cloned']} cloned, "
                        f"+{stats['split']*2} split, -{stats['pruned']} pruned "
                        f"= {stats['total']} total"
                    )

            if stage_name == "C_detail" and local_iter > 0 and local_iter % 3000 == 0:
                model.reset_opacities()
                tqdm.write(f"[{global_iter}] Reset opacities")

            if local_iter % args.log_interval == 0:
                elapsed = time.time() - start_time
                xi_norm = cameras.xi.data.norm(dim=-1).mean().item()
                log_entry = {
                    "iter": global_iter,
                    "stage": stage_name,
                    "loss": loss.item(),
                    "gaussians": model.num_gaussians,
                    "xi_norm": xi_norm,
                    "elapsed": elapsed,
                }
                log["iterations"].append(log_entry)
                if local_iter % (args.log_interval * 10) == 0:
                    tqdm.write(
                        f"[{global_iter}] loss={loss.item():.4f} "
                        f"n={model.num_gaussians} xi={xi_norm:.4f}"
                    )

            if local_iter % args.save_interval == 0 and global_iter > 0:
                save_snapshot(
                    rendered, target_images[cam_idx], global_iter, stage_name,
                    loss.item(), model.num_gaussians,
                    os.path.join(args.output, "snapshots"),
                )

            global_iter += 1

        log["stages"].append({"name": stage_name, "final_iter": global_iter})

    # Done
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s ({global_iter/elapsed:.1f} it/s)")

    # Export
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

    # Save camera poses as JSON
    cameras.reanchor()
    viewmats = cameras.viewmats().detach().cpu()
    print(f"Exporting {num_cameras} cameras (viewmats shape: {viewmats.shape})")
    cam_data = []
    for i in range(num_cameras):
        R = viewmats[i, :3, :3]
        t = viewmats[i, :3, 3]
        q = rotation_matrix_to_quaternion(R)
        cam_data.append({
            "quat": q.tolist(),
            "trans": t.tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height,
        })

    cam_path = os.path.join(args.output, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cam_data, f, indent=2)
    print(f"Saved camera poses to {cam_path}")

    # Save training log
    log_path = os.path.join(args.output, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved training log to {log_path}")


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert [3,3] rotation matrix to [w,x,y,z] quaternion. Numerically robust."""
    diag = torch.stack([R[0,0], R[1,1], R[2,2], R.trace()])
    idx = diag.argmax().item()
    if idx == 3:
        s = (1.0 + R.trace()).clamp(min=1e-8).sqrt() * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif idx == 0:
        s = (1.0 + R[0,0] - R[1,1] - R[2,2]).clamp(min=1e-8).sqrt() * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif idx == 1:
        s = (1.0 + R[1,1] - R[0,0] - R[2,2]).clamp(min=1e-8).sqrt() * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = (1.0 + R[2,2] - R[0,0] - R[1,1]).clamp(min=1e-8).sqrt() * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return torch.tensor([w, x, y, z])


def save_snapshot(
    rendered: torch.Tensor,
    target: torch.Tensor,
    iteration: int,
    stage: str,
    loss_val: float,
    num_gaussians: int,
    output_dir: str,
):
    import cv2
    os.makedirs(output_dir, exist_ok=True)

    r = (rendered.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    t = (target.detach().cpu().clamp(0, 1) * 255).byte().numpy()

    h, w = r.shape[:2]
    canvas = np.zeros((h + 30, w * 2 + 4, 3), dtype=np.uint8)
    canvas[:h, :w] = r
    canvas[:h, w+4:] = t

    label = f"iter {iteration} | {stage} | loss {loss_val:.4f} | {num_gaussians} splats"
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas_bgr, label, (8, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas_bgr, "rendered", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(canvas_bgr, "target", (w + 12, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    path = os.path.join(output_dir, f"snapshot_{iteration:06d}.png")
    cv2.imwrite(path, canvas_bgr)


def main():
    parser = argparse.ArgumentParser(description="COLMAP-free 3D Gaussian Splatting")
    parser.add_argument("--input", required=True, help="Video file or image directory")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to extract from video")
    parser.add_argument("--num-gaussians", type=int, default=100000, help="Initial gaussian count")
    parser.add_argument("--lambda-ssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500, help="Save snapshot every N iterations")
    parser.add_argument("--stages", default="A,B,C", help="Comma-separated stages to run: A (warmup), B (joint), C (detail)")
    parser.add_argument("--views-per-iter", type=int, default=3, help="Number of views rendered per iteration for 3D consistency")
    args = parser.parse_args()

    if rasterization is None:
        print("ERROR: gsplat is not installed. Install with: pip install gsplat")
        print("Requires CUDA GPU. See https://github.com/nerfstudio-project/gsplat")
        return

    train(args)


if __name__ == "__main__":
    main()
