# Gaussian Splat Studio — Plan

## Context

We want a Mac app that takes a video (or series of images) and constructs a 3D Gaussian Splat model. The key novelty: **no COLMAP/SfM/pointcloud step**. Instead, we start with randomized splats and camera positions, then jointly optimize everything via gradient descent. Camera gradients are obtained by the "move splats the opposite direction" trick — transforming splats into camera space before rasterizing, so the existing position gradients give camera gradients for free.

## Key Research Finding

**gsplat** (nerfstudio-project/gsplat) already separates camera transforms from rasterization exactly the way we want:
- `pos_cam = R @ pos_world + t` happens outside the rasterizer
- Backward pass gives `dL/d(pos_cam)` per splat
- `dL/dt = sum(dL/d(pos_cam))` and `dL/dR` chains through position + covariance transforms
- No rasterizer modification needed — we wrap the existing one

Other useful repos:
- **splat-apple** (ghif/splat-apple) — MLX/Metal training, full backward pass, ~150 lines of pure-array rasterizer
- **MetalSplatter** (scier/MetalSplatter) — Swift/Metal viewer for .ply/.splat files
- **diff-gaussian-rasterization** (graphdeco-inria) — reference CUDA rasterizer (camera baked in, harder to use)
- **rmurai0610/diff-gaussian-rasterization-w-pose** — fork with SE(3) camera pose gradients

## Training Cost Estimates

### CUDA GPU training times (30,000 iterations)
| GPU | Time | Cloud cost (Vast.ai) |
|---|---|---|
| RTX 4090 | ~12 min | ~$0.06 |
| RTX 3090 | ~20 min | ~$0.04 |
| A100 80GB | ~10 min | ~$0.30 |
| H100 | ~8 min | ~$0.36 |
| FastGS (CVPR 2026) | 77–100 sec | ~$0.007 |

COLMAP-free joint camera optimization adds **negligible overhead** (0–5% extra time per papers: GloSplat, 3R-GS, TrackGS).

### Apple Silicon estimates
No published benchmarks for splat-apple iteration speed. General expectation: **10–50x slower** than CUDA GPUs for ML training workloads. Rough estimate: ~2–10 hours on M3 Max for 30k iterations. Needs benchmarking.

### Memory
- ~24 GB VRAM for standard training (up to ~5M gaussians)
- gsplat uses 4x less memory than the original implementation
- Apple Silicon unified memory is an advantage here — 64–128GB Macs can handle large scenes

### Cloud provider: RunPod
RunPod is suitable. RTX 4090 at $0.34–0.59/hr. A single training run costs pennies. Already have API key.

### Mobile feasibility
- **Rendering on phone**: YES — Mobile-GS achieves 116 FPS on Snapdragon 8 Gen 3 at 1080p with 4.8 MB model
- **Training on phone**: NO (practical) — too slow, too memory-constrained. VBGS did it on Jetson Orin Nano but still far slower than GPU
- **Best mobile UX**: capture video on phone → upload to cloud/Mac backend → train → download compressed .ply → render locally on phone

## Architecture

Three deliverables:

1. **Training backend** — Python + gsplat, runs on RunPod (CUDA) or local Apple Silicon
2. **Mac viewer app** — SwiftUI + MetalSplatter, view trained .ply models with orbit controls
3. **Android capture + viewer app** — capture video/photos, upload to backend, download .ply, render locally

### System flow
```
[Android phone]                    [RunPod / Mac]              [Mac / Android]
 capture video  ──upload──>  training backend  ──.ply──>   viewer app
                              (Python + gsplat)             (Metal / OpenGL ES)
```

### Cloud backend (RunPod)
- API key: (set RUNPOD_API_KEY env var)
- Use serverless or on-demand pod with RTX 4090
- Accept video upload → extract frames → train → return .ply
- A training run costs ~$0.06

### Test videos (received via LocalSend, descending difficulty)
1. `VID_20260503_125827983.mp4` — 17s, 1080p (hardest)
2. `VID_20260503_125858533.mp4` — 25s, 1080p
3. `VID_20260503_125941184.mp4` — 24s, 1080p (easiest)

## Implementation Steps

### Step 1: Create repo and project structure

Create `gaussian_splat` GitHub repo, add `kim-em` as collaborator.

```
gaussian_splat/
├── README.md                    # Project overview, setup instructions
├── PLAN.md                      # This plan (detailed)
├── training/
│   ├── requirements.txt         # gsplat, torch, etc.
│   ├── train.py                 # Main training script
│   ├── gaussian_model.py        # Gaussian parameter management
│   ├── camera_model.py          # Camera parameters + transform
│   ├── densification.py         # Clone / split / prune
│   ├── loss.py                  # L1 + SSIM
│   └── utils.py                 # Frame extraction, I/O
├── viewer/                      # (future) Swift Mac app
└── test/
    ├── test_camera_gradients.py # Numerical gradient verification
    └── test_densification.py
```

### Step 2: Camera model with "move splats opposite" approach

`camera_model.py` — each camera stores a quaternion (4) + translation (3):

```python
def transform_to_camera(means3D, quats_world, scales, camera_q, camera_t):
    """Transform all splats into camera space.
    
    This is the key trick: instead of differentiating the rasterizer
    w.r.t. camera params, we move all splats by the inverse camera
    transform. The rasterizer's existing position gradients then
    give us camera gradients for free:
      dL/d(camera_t) = -sum(dL/d(pos_cam))
      dL/d(camera_R) chains through pos_cam and cov_cam
    """
    R = quaternion_to_matrix(camera_q)        # [3,3]
    means_cam = (means3D @ R.T) + camera_t    # [N, 3]
    # Also rotate covariances: cov_cam = R @ cov_world @ R^T
    # Also rotate splat quaternions into camera frame
    return means_cam, rotated_quats, scales
```

Camera params are `torch.nn.Parameter` with `requires_grad=True`. Gradients flow automatically through `transform_to_camera` into `camera_q` and `camera_t`.

### Step 3: Training loop with joint optimization

`train.py`:

```python
for iteration in range(30_000):
    # 1. Pick random training view
    idx = random.randint(0, num_cameras - 1)
    
    # 2. Transform splats into this camera's frame
    means_cam, quats_cam, scales = transform_to_camera(
        gaussians.means, gaussians.quats, gaussians.scales,
        cameras[idx].q, cameras[idx].t
    )
    
    # 3. Rasterize with identity camera (gsplat)
    rendered = rasterize(means_cam, quats_cam, scales, 
                         gaussians.opacities, gaussians.sh_coeffs,
                         identity_viewmat, intrinsics)
    
    # 4. Loss
    loss = (1 - λ) * l1_loss(rendered, target[idx]) + λ * (1 - ssim(rendered, target[idx]))
    
    # 5. Backward — gradients flow to both gaussian params AND camera params
    loss.backward()
    
    # 6. Update
    gaussian_optimizer.step()    # Adam, per-param LR
    camera_optimizer.step()      # Adam, lower LR
    
    # 7. Densification (every 100 iters, after iter 500, before iter 15000)
    if 500 <= iteration < 15000 and iteration % 100 == 0:
        densify(gaussians, grad_accum)
```

**Optimizer LRs:**
- Splat positions: 1.6e-4 → 1.6e-6 (exponential decay)
- Splat scales: 5e-3
- Splat rotations: 1e-3
- Splat opacity: 5e-2
- Splat SH: 2.5e-3
- Camera quaternion: 1e-4
- Camera translation: 1e-3

### Step 4: Densification (clone / split / prune)

`densification.py` — runs every 100 iterations from iter 500 to 15000:

1. **Track gradients**: accumulate `||dL/d(mean)||` per splat over 100 iterations
2. **Clone**: splats with large avg gradient AND small scale → duplicate as-is
3. **Split**: splats with large avg gradient AND large scale → replace with 2 half-scale splats offset along principal axis
4. **Prune**: remove splats with opacity < 0.005, or scale > 10% of scene extent
5. **Opacity reset**: every 3000 iterations, reset all opacities to 0.01 (prevents floaters)
6. **Reset optimizer state** for new/modified splats

### Step 5: Input pipeline

`utils.py`:
- **Video**: extract frames with OpenCV or ffmpeg, subsample to ~100-300 frames
- **Images**: load from directory
- Resize to training resolution (start 400×400, optionally progressive upscale)
- Estimate intrinsics: `fx = fy = max(W, H)`, `cx = W/2`, `cy = H/2`

### Step 6: Initialization strategy (no COLMAP)

Since we have no prior camera poses:
1. **Gaussians**: random positions in [-1, 1]³, random colors, small uniform scale, full opacity. Start with ~10,000.
2. **Cameras**: distribute on a hemisphere of radius 3, looking at origin, with small random perturbations. For video input, arrange sequentially along a smooth arc.
3. **Warm-up phase** (iters 0–2000): higher camera LR, coarse resolution (200×200), fewer gaussians. Goal: find approximate camera arrangement.
4. **Joint phase** (iters 2000–30000): normal LRs, full resolution, densification active.

### Step 7: Export and verification

- Export trained gaussians to .ply (standard format)
- Export recovered camera poses
- Test with a known synthetic scene (e.g., render a cube from known cameras, verify recovery)

### Step 8: Mac viewer app

SwiftUI + Metal app for viewing trained gaussian splat models.

- **Dependency**: MetalSplatter (scier/MetalSplatter) — Swift/Metal .ply/.splat renderer
- **Features**:
  - Open .ply files from training output
  - Orbit camera controls (drag to rotate, pinch to zoom, pan)
  - Show recovered camera positions as frustum wireframes
  - Side-by-side: original video frames vs rendered views from recovered cameras
  - Basic model info: gaussian count, bounding box, file size
- **Build**: Xcode project, SPM dependency on MetalSplatter

### Step 9: Android capture + viewer app

Kotlin/Jetpack Compose app for capturing video and viewing results.

- **Capture flow**:
  - Record video or select existing video/photos
  - Upload to training backend (RunPod endpoint or local Mac server)
  - Show training progress (poll for status)
  - Download .ply when done
- **Viewer**:
  - OpenGL ES gaussian splat renderer (port Mobile-GS approach, or use WebView with three.js viewer)
  - Touch controls: rotate, zoom, pan
  - Save/load models locally
- **Backend communication**: REST API (upload video, poll status, download result)
- **Build**: `./gradlew assembleDebug`, sideload to phone (ZY22KCP5SQ)

### Step 10: Training backend API

Simple Flask/FastAPI server wrapping the training script, deployable to RunPod.

```
POST /train          — upload video, returns job_id
GET  /status/{id}    — training progress (iteration, loss, ETA)
GET  /result/{id}    — download .ply when done
```

- Dockerized for RunPod deployment
- Stores intermediate renders for progress preview
- Cleans up after download

## Verification

1. **Camera gradient test**: create a synthetic scene with known camera poses. Perturb cameras slightly. Verify that `loss.backward()` produces gradients that move cameras back toward ground truth. Compare autodiff gradients with finite-difference numerical gradients.
2. **Convergence test on synthetic data**: render a simple scene (colored cubes) from known cameras. Start from random cameras + random splats. Verify both cameras and splats converge.
3. **Real video test**: take a short video of an object, run training, inspect .ply output in a viewer.

## Repo Structure (updated)

```
gaussian_splat/
├── README.md
├── PLAN.md
├── training/
│   ├── Dockerfile               # RunPod deployment
│   ├── requirements.txt         # gsplat, torch, flask, etc.
│   ├── train.py                 # Main training script
│   ├── server.py                # REST API for remote training
│   ├── gaussian_model.py
│   ├── camera_model.py
│   ├── densification.py
│   ├── loss.py
│   └── utils.py
├── mac-viewer/                  # Xcode SwiftUI project
│   ├── Package.swift            # SPM deps (MetalSplatter)
│   └── GaussianViewer/
│       ├── GaussianViewerApp.swift
│       ├── ContentView.swift
│       ├── SplatView.swift      # MetalSplatter wrapper
│       └── CameraFrustumView.swift
├── android-app/                 # Gradle project
│   └── app/src/main/
│       ├── java/.../
│       │   ├── MainActivity.kt
│       │   ├── CaptureScreen.kt
│       │   ├── TrainingScreen.kt
│       │   └── ViewerScreen.kt
│       └── res/
├── test/
│   ├── test_camera_gradients.py
│   └── test_densification.py
└── test_videos/                 # Symlinks or notes pointing to test data
```

## Milestones

1. **M1: Training works on synthetic data** — camera gradient test passes, convergence on known scene
2. **M2: Training works on real video** — run on test videos, produce viewable .ply
3. **M3: RunPod deployment** — Dockerized, API endpoint, can submit jobs remotely
4. **M4: Mac viewer** — open .ply, orbit camera, show recovered cameras
5. **M5: Android app** — capture video, upload, view result

## Immediate Actions

- [ ] Create GitHub repo `gaussian_splat`
- [ ] Add `kim-em` as collaborator
- [ ] Commit PLAN.md + initial project structure
- [ ] Implement training core (steps 2–4)
- [ ] Test with synthetic scene
- [ ] Test with the 3 LocalSend videos
- [ ] Deploy to RunPod
- [ ] Build Mac viewer
- [ ] Build Android app
