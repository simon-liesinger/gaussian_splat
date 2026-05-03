#!/usr/bin/env python3
"""Run SfM on all datasets locally, save results for RunPod training.

Extracts frames from videos, runs pycolmap SfM, saves sfm_result.json + frames.
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
from sfm import run_sfm

DATASETS = {
    "hairbrush": {
        "input": "/Users/simon/Downloads/VID_20260503_125941184.mp4",
        "type": "video",
        "max_frames": 300,
        "sfm_frames": 50,
    },
    "video2_medium": {
        "input": "/Users/simon/Downloads/VID_20260503_125858533.mp4",
        "type": "video",
        "max_frames": 300,
        "sfm_frames": 50,
    },
    "video3_hard": {
        "input": "/Users/simon/Downloads/VID_20260503_125827983.mp4",
        "type": "video",
        "max_frames": 200,
        "sfm_frames": 50,
    },
    "phone_images": {
        "input": "/Users/simon/claude/gaussian_splat/phone_images",
        "type": "images",
        "max_frames": None,  # use all
        "sfm_frames": None,  # use all for SfM
    },
}


def extract_video_frames(video_path, output_dir, max_frames=300, sfm_subset=50):
    """Extract frames from video. Returns (all_frames_dir, sfm_frames_dir)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract all frames
    all_dir = os.path.join(output_dir, "all_frames")
    os.makedirs(all_dir, exist_ok=True)
    step = max(1, total // max_frames)
    count = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(all_dir, f"frame_{count:04d}.jpg"), frame)
            count += 1
    cap.release()
    print(f"  Extracted {count} frames to {all_dir}")

    # Create SfM subset (evenly spaced from the full set)
    sfm_dir = os.path.join(output_dir, "sfm_frames")
    os.makedirs(sfm_dir, exist_ok=True)
    all_files = sorted(Path(all_dir).glob("*.jpg"))
    sfm_step = max(1, len(all_files) // sfm_subset)
    sfm_count = 0
    for i in range(0, len(all_files), sfm_step):
        src = all_files[i]
        dst = os.path.join(sfm_dir, src.name)
        shutil.copy2(str(src), dst)
        sfm_count += 1
    print(f"  SfM subset: {sfm_count} frames in {sfm_dir}")

    return all_dir, sfm_dir


def prepare_image_dir(image_dir, output_dir):
    """Prepare image directory (filter ._ files). Returns (all_dir, sfm_dir)."""
    all_dir = os.path.join(output_dir, "all_frames")
    os.makedirs(all_dir, exist_ok=True)
    count = 0
    for f in sorted(Path(image_dir).glob("*.jpg")):
        if f.name.startswith("._"):
            continue
        dst = os.path.join(all_dir, f.name)
        shutil.copy2(str(f), dst)
        count += 1
    print(f"  Linked {count} images to {all_dir}")
    # Use all images for SfM
    return all_dir, all_dir


def main():
    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for name in datasets_to_run:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        ds = DATASETS[name]
        output_dir = os.path.join("sfm_data", name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        # Check if already done
        result_path = os.path.join(output_dir, "sfm_result.json")
        if os.path.exists(result_path):
            print(f"  Already done, skipping. Delete {result_path} to rerun.")
            continue

        # Extract/prepare frames
        if ds["type"] == "video":
            if not os.path.exists(ds["input"]):
                print(f"  Video not found: {ds['input']}, skipping")
                continue
            all_dir, sfm_dir = extract_video_frames(
                ds["input"], output_dir,
                max_frames=ds["max_frames"],
                sfm_subset=ds["sfm_frames"],
            )
        else:
            if not os.path.exists(ds["input"]):
                print(f"  Image dir not found: {ds['input']}, skipping")
                continue
            all_dir, sfm_dir = prepare_image_dir(ds["input"], output_dir)

        # Run SfM
        print(f"  Running SfM on {sfm_dir}...")
        try:
            result = run_sfm(sfm_dir)
        except Exception as e:
            print(f"  SfM FAILED: {e}")
            continue

        print(f"  SfM result: {len(result['cameras'])} cameras, {len(result['points3d'])} points")

        # Save result
        data = {
            "cameras": [c.tolist() for c in result["cameras"]],
            "intrinsics": result["intrinsics"],
            "image_names": result["image_names"],
            "points3d": result["points3d"].tolist(),
            "point_colors": result["point_colors"].tolist(),
            "all_frames_dir": all_dir,
        }
        with open(result_path, "w") as f:
            json.dump(data, f)
        print(f"  Saved {result_path}")

        # Create tarball for RunPod upload
        import subprocess
        tar_path = os.path.join(output_dir, "upload.tar.gz")
        subprocess.run(
            ["tar", "czf", tar_path, "-C", output_dir, "all_frames", "sfm_result.json"],
            check=True,
        )
        size_mb = os.path.getsize(tar_path) / 1e6
        print(f"  Created {tar_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
