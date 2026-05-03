"""Structure-from-Motion via pycolmap.

Extracts camera poses and sparse point cloud from a set of images.
Used to initialize gaussian splats with known camera positions
instead of cold-start random initialization.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def run_sfm(image_dir: str, device: torch.device = torch.device("cuda")) -> dict:
    """Run COLMAP SfM on a directory of images.

    Returns:
        dict with keys:
            "cameras": list of [4, 4] world-to-camera matrices
            "intrinsics": list of (fx, fy, cx, cy) per camera
            "image_names": list of filenames in order
            "points3d": [N, 3] numpy array of sparse 3D points
            "point_colors": [N, 3] numpy array of point RGB colors (0-255)
    """
    import pycolmap

    image_dir = str(image_dir)
    work_dir = tempfile.mkdtemp(prefix="colmap_")
    db_path = os.path.join(work_dir, "database.db")

    print(f"Running SfM on {image_dir}...")
    print(f"  Work dir: {work_dir}")

    # Feature extraction
    print("  Extracting features...")
    pycolmap.extract_features(db_path, image_dir)

    # Feature matching
    print("  Matching features...")
    pycolmap.match_exhaustive(db_path)

    # Incremental SfM
    print("  Running incremental mapper...")
    output_dir = os.path.join(work_dir, "sparse")
    os.makedirs(output_dir, exist_ok=True)
    maps = pycolmap.incremental_mapping(db_path, image_dir, output_dir)

    if not maps:
        raise RuntimeError("SfM failed: no reconstruction produced")

    # Use the largest reconstruction
    rec = maps[0]
    print(f"  Reconstruction: {rec.num_images()} images, {rec.num_points3D()} points")

    # Extract camera poses
    cameras_out = []
    intrinsics_out = []
    image_names = []

    # Sort by image name for consistent ordering
    sorted_images = sorted(rec.images.values(), key=lambda img: img.name)

    for image in sorted_images:
        # World-to-camera transform
        # pycolmap 4.x: cam_from_world() is a method returning Rigid3d
        cam_from_world = image.cam_from_world()
        R = np.array(cam_from_world.rotation.matrix())  # 3x3
        t = np.array(cam_from_world.translation)  # 3-vector

        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        cameras_out.append(viewmat)

        # Intrinsics
        cam = rec.cameras[image.camera_id]
        params = cam.params
        if cam.model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam.model_name in ("PINHOLE", "OPENCV"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            # Fallback: assume first param is focal
            fx = fy = params[0]
            cx = cam.width / 2
            cy = cam.height / 2
        intrinsics_out.append((fx, fy, cx, cy))

        image_names.append(image.name)

    # Extract 3D points
    points = []
    colors = []
    for point in rec.points3D.values():
        points.append(point.xyz)
        colors.append(point.color)

    points3d = np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
    point_colors = np.array(colors, dtype=np.uint8) if colors else np.zeros((0, 3), dtype=np.uint8)

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"  SfM complete: {len(cameras_out)} cameras, {len(points3d)} points")

    return {
        "cameras": cameras_out,
        "intrinsics": intrinsics_out,
        "image_names": image_names,
        "points3d": points3d,
        "point_colors": point_colors,
    }
