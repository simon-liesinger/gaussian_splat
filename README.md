# Gaussian Splat Studio

COLMAP-free 3D Gaussian Splatting with joint camera pose optimization.

Takes a video or series of images and constructs a 3D gaussian splat model **from scratch** -- no COLMAP, no Structure-from-Motion, no pointcloud preprocessing. Camera positions and splat parameters are jointly optimized via gradient descent.

## How it works

The key trick: instead of modifying the rasterizer to differentiate w.r.t. camera parameters, we transform all splats into camera space before rasterizing. The rasterizer's existing position gradients then flow back through the transform into camera quaternion and translation parameters.

```
pos_cam = R(camera_q) @ pos_world + camera_t
```

Since this is standard PyTorch math, `loss.backward()` automatically gives us `dL/d(camera_q)` and `dL/d(camera_t)`.

Uses [gsplat](https://github.com/nerfstudio-project/gsplat) as the differentiable rasterizer (unmodified).

## Components

- **training/** -- Python training backend (PyTorch + gsplat)
- **mac-viewer/** -- SwiftUI + Metal viewer app (planned)
- **android-app/** -- Android capture + viewer app (planned)

## Quick start

```bash
cd training
pip install -r requirements.txt
python train.py --input video.mp4 --output output/
```

Requires a CUDA GPU (or RunPod cloud instance). See PLAN.md for full details.

## Test

```bash
cd test
python test_camera_gradients.py
```
