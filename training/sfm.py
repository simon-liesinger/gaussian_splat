"""Structure-from-Motion via pycolmap.

Extracts camera poses and sparse point cloud from a set of images.
Supports two-phase pipeline: full SfM on a subset, then cheap registration
of additional images against the existing reconstruction.
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
            "cameras": list of [4, 4] numpy arrays (world-to-camera matrices)
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

    # Feature matching — sequential for video (>50 images), exhaustive for small sets
    num_images = len([f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._')])
    if num_images > 50:
        print(f"  Matching features (sequential, {num_images} images)...")
        pycolmap.match_sequential(db_path)
    else:
        print(f"  Matching features (exhaustive, {num_images} images)...")
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

    result = _extract_reconstruction(rec)

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"  SfM complete: {len(result['cameras'])} cameras, {len(result['points3d'])} points")
    return result


def run_sfm_with_registration(
    initial_image_dir: str,
    extra_image_dir: str,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Two-phase SfM: full reconstruction on initial images, then register extras.

    Args:
        initial_image_dir: directory with subset of images for full SfM
        extra_image_dir: directory with ALL images (initial + extra) to register

    Returns:
        Same format as run_sfm(), with all successfully registered cameras.
    """
    import pycolmap

    work_dir = tempfile.mkdtemp(prefix="colmap_reg_")
    db_path = os.path.join(work_dir, "database.db")

    # Phase 1: Full SfM on initial subset
    print(f"Phase 1: Full SfM on {initial_image_dir}...")
    pycolmap.extract_features(db_path, initial_image_dir)

    num_initial = len([f for f in os.listdir(initial_image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._')])
    if num_initial > 50:
        print(f"  Sequential matching ({num_initial} images)...")
        pycolmap.match_sequential(db_path)
    else:
        print(f"  Exhaustive matching ({num_initial} images)...")
        pycolmap.match_exhaustive(db_path)

    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    maps = pycolmap.incremental_mapping(db_path, initial_image_dir, sparse_dir)

    if not maps:
        raise RuntimeError("Phase 1 SfM failed")

    rec = maps[0]
    print(f"  Phase 1: {rec.num_images()} images, {rec.num_points3D()} points")

    # Phase 2: Extract features for extra images and register them
    print(f"Phase 2: Registering extra images from {extra_image_dir}...")

    # Create a new database with all images
    db_path_all = os.path.join(work_dir, "database_all.db")
    pycolmap.extract_features(db_path_all, extra_image_dir)

    # Match all images against each other (sequential for video)
    num_all = len([f for f in os.listdir(extra_image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._')])
    if num_all > 50:
        print(f"  Sequential matching ({num_all} images)...")
        pycolmap.match_sequential(db_path_all)
    else:
        print(f"  Exhaustive matching ({num_all} images)...")
        pycolmap.match_exhaustive(db_path_all)

    # Run incremental mapping on all images, using the existing model as starting point
    sparse_dir_all = os.path.join(work_dir, "sparse_all")
    os.makedirs(sparse_dir_all, exist_ok=True)

    # Save the initial reconstruction for the mapper to use
    input_sparse = os.path.join(work_dir, "sparse", "0")
    rec.write(input_sparse)

    # Run mapper on all images — it will use the existing model as prior
    maps_all = pycolmap.incremental_mapping(
        db_path_all, extra_image_dir, sparse_dir_all,
    )

    if maps_all:
        rec_all = maps_all[0]
        print(f"  Phase 2: {rec_all.num_images()} images, {rec_all.num_points3D()} points")
        result = _extract_reconstruction(rec_all)
    else:
        print(f"  Phase 2 registration failed, using Phase 1 results only")
        result = _extract_reconstruction(rec)

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"  Total: {len(result['cameras'])} cameras, {len(result['points3d'])} points")
    return result


def _extract_reconstruction(rec) -> dict:
    """Extract camera poses and points from a pycolmap Reconstruction."""
    cameras_out = []
    intrinsics_out = []
    image_names = []

    sorted_images = sorted(rec.images.values(), key=lambda img: img.name)

    for image in sorted_images:
        cam_from_world = image.cam_from_world()
        R = np.array(cam_from_world.rotation.matrix())
        t = np.array(cam_from_world.translation)

        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        cameras_out.append(viewmat)

        cam = rec.cameras[image.camera_id]
        params = cam.params
        if cam.model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam.model_name in ("PINHOLE", "OPENCV"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            fx = fy = params[0]
            cx = cam.width / 2
            cy = cam.height / 2
        intrinsics_out.append((fx, fy, cx, cy))
        image_names.append(image.name)

    points = []
    colors = []
    for point in rec.points3D.values():
        points.append(point.xyz)
        colors.append(point.color)

    points3d = np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
    point_colors = np.array(colors, dtype=np.uint8) if colors else np.zeros((0, 3), dtype=np.uint8)

    return {
        "cameras": cameras_out,
        "intrinsics": intrinsics_out,
        "image_names": image_names,
        "points3d": points3d,
        "point_colors": point_colors,
    }
