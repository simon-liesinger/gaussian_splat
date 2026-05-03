# Gaussian Splat Studio — Plan

## Context

We want a Mac app that takes a video (or series of images) and constructs a 3D Gaussian Splat model. The deliberate constraint: **no COLMAP, no SfM, no learned pose prior, no depth network**. We start from random splats and (almost-)random cameras and let joint photometric gradient descent figure everything out.

This is unproven. Every "COLMAP-free" paper we know of (3R-GS, TrackGS, GloSplat) actually uses a strong prior — MASt3R-SfM, learned 3D tracks, or global feature-based SfM — and only refines poses jointly with appearance. We're going further: true cold start. It might not work on real handheld video. **That's the experiment.** The plan below tries to give cold start the best possible chance and to fail informatively when it doesn't.

## How Camera Gradients Work

**gsplat already supports differentiable camera poses natively.** `gsplat.rasterization(...)` accepts `viewmats` as a `[C, 4, 4]` tensor; PyTorch autograd flows through it. Setting `viewmats.requires_grad_(True)` is sufficient — no rasterizer modification, no wrapper. Intrinsics `Ks` are not differentiable, so we keep them fixed.

An earlier version of this plan proposed a "transform splats into camera space, rasterize with identity" wrapper to extract camera gradients via splat position gradients. That wrapper is mathematically equivalent to what gsplat does internally and is therefore redundant. The current code in `training/camera_model.py` still implements it; we will replace it with direct `viewmats` optimization.

**Pose parameterization.** Storing a raw quaternion + translation and renormalizing each step is brittle for cold-start optimization where poses move large distances. We will parameterize each camera as `(R0, t0)` (a fixed reference pose) plus a learnable `se(3)` tangent vector `ξ ∈ R^6`, applied via the matrix exponential. This gives a well-conditioned local update without unit-norm drift, and lets us re-anchor `(R0, t0)` periodically if `ξ` grows large.

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

**Caveat on the 0–5% overhead claim.** That number from GloSplat/3R-GS/TrackGS measures *per-iteration* pose-refinement cost on top of a strong prior. It does **not** measure end-to-end cold-start cost. Cold start may need many more iterations, may converge slowly, or may fail to converge for non-object-centric captures. Treat 30k iterations as a starting point, not a budget.

### Apple Silicon estimates — BENCHMARK FIRST

No published benchmarks for gsplat-on-MPS or splat-apple iteration speed. Rough expectation is 10–50× slower than CUDA, putting 30k iterations somewhere in the 2–10 hour range on an M3 Max — but this is a guess with a 5× spread.

**This needs to be the first thing we measure.** It determines whether the inner dev loop is local (great) or RunPod-only (acceptable but slower iteration). Concrete benchmark target: time per iteration at 100k splats × {200², 400², 800²} resolution, with both `mps` and `cpu` backends, on the available Apple Silicon hardware. See M0 below.

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

### Step 2: Camera model — direct viewmat optimization in SE(3)

`camera_model.py` — each camera stores a fixed reference pose `(R0, t0)` plus a learnable `se(3)` tangent vector `ξ ∈ R^6` = `(ω, v)` (rotation + translation parts of the Lie algebra).

```python
class CameraSet(nn.Module):
    def __init__(self, num_cameras: int, init_poses: Tensor):
        # init_poses: [N, 4, 4] world-to-camera matrices from initialization
        self.register_buffer("R0", init_poses[:, :3, :3])  # [N, 3, 3]
        self.register_buffer("t0", init_poses[:, :3, 3])   # [N, 3]
        self.xi = nn.Parameter(torch.zeros(num_cameras, 6))  # se(3) tangent

    def viewmats(self) -> Tensor:
        """Return [N, 4, 4] world-to-camera matrices, differentiable in self.xi."""
        delta = se3_exp(self.xi)              # [N, 4, 4]
        base = pose_to_matrix(self.R0, self.t0)  # [N, 4, 4]
        return delta @ base                   # left-multiply: refine in camera frame

    def reanchor(self):
        """Fold xi into (R0, t0) and reset xi to zero. Call when ||xi|| grows."""
        with torch.no_grad():
            new = self.viewmats()
            self.R0.copy_(new[:, :3, :3])
            self.t0.copy_(new[:, :3, 3])
            self.xi.zero_()
```

Then in the training loop we just hand `cameras.viewmats()` to gsplat:

```python
out = gsplat.rasterization(
    means=gaussians.means, quats=gaussians.quats,
    scales=gaussians.scales, opacities=gaussians.opacities,
    colors=gaussians.colors,
    viewmats=cameras.viewmats(),         # autograd flows here
    Ks=intrinsics, width=W, height=H,
)
```

No splat transform wrapper. Gradients to `cameras.xi` come for free via gsplat's existing `viewmats` differentiability. This is the **replacement** for the current `training/camera_model.py`, which still uses the wrapper approach and needs to be rewritten.

### Step 3: Training loop with joint optimization

`train.py`:

```python
for iteration in range(num_iters):
    # 1. Pick a batch of training views
    idxs = sample_view_batch(...)

    # 2. Render — gradients flow into gaussians AND cameras.xi
    rendered, _, _ = gsplat.rasterization(
        means=gaussians.means, quats=gaussians.quats,
        scales=gaussians.scales, opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=cameras.viewmats()[idxs],
        Ks=intrinsics[idxs], width=W, height=H,
    )

    # 3. Loss
    loss = (1 - λ) * l1(rendered, target[idxs]) \
         + λ * (1 - ssim(rendered, target[idxs]))

    # 4. Backward
    loss.backward()
    gaussian_optimizer.step()
    camera_optimizer.step()

    # 5. Periodically re-anchor camera Lie-algebra tangents to keep them small
    if iteration % 500 == 0:
        cameras.reanchor()

    # 6. Densification — see Step 4. Disabled until pose stabilizes.
    if densification_active(iteration, pose_change_rate):
        densify(gaussians, grad_accum)
```

**Optimizer LRs (starting points; will need tuning under cold start):**
- Splat positions: 1.6e-4 → 1.6e-6 (exponential decay, cosine over training)
- Splat scales: 5e-3
- Splat rotations: 1e-3
- Splat opacity: 5e-2
- Splat colors / SH: 2.5e-3
- Camera `ξ` rotation part: 1e-3 during warm-up, decay to 1e-5
- Camera `ξ` translation part: 1e-2 during warm-up, decay to 1e-4

Cold start needs **higher** camera LRs early (poses move large distances) and aggressive decay later (avoid jitter once roughly converged). This is the opposite of the standard 3DGS pose-refinement schedule, which assumes COLMAP poses are nearly correct.

### Step 4: Densification (clone / split / prune)

The standard 3DGS schedule (start at iter 500, end at 15000, opacity reset every 3000) assumes COLMAP-initialized splats already sitting near the right geometry. Under cold start, the early splats are random and the early poses are wrong, so `||dL/d(mean)||` is dominated by pose error rather than missing geometry. Densifying on that signal will amplify garbage.

`densification.py` — gated on pose stability:

1. **Pose stability gate**: track running mean of `||Δξ||` per camera over the last N iterations. Only enable densification once the mean drops below a threshold (geometry has stopped chasing poses). For cold start this is likely 5–10k iterations in, not 500.
2. **Track gradients**: accumulate `||dL/d(mean)||` per splat over 100 iterations once active.
3. **Clone**: splats with large avg gradient AND small scale → duplicate as-is.
4. **Split**: splats with large avg gradient AND large scale → replace with 2 half-scale splats offset along principal axis.
5. **Prune**: remove splats with opacity < 0.005, or scale > 10% of scene extent.
6. **Opacity reset**: only after densification has been running for several thousand iterations; not during pose convergence (would erase whatever geometry the poses have started to lock onto).
7. **Reset optimizer state** for new/modified splats.

Schedule is intentionally adaptive rather than hardcoded. We will instrument and tune.

### Step 5: Input pipeline

`utils.py`:
- **Video**: extract frames with OpenCV or ffmpeg, subsample to ~100-300 frames
- **Images**: load from directory
- Resize to training resolution (start 200×200, progressive upscale to 400², 800², full)
- **Intrinsics**: read EXIF focal length and sensor size when available; convert to pixel focal length. Fall back to `fx = fy = 1.2 × max(W, H)` (rough phone-camera prior) only if EXIF is missing. Keep intrinsics **fixed** through training — adding focal-length optimization to cold-start pose+geometry adds a third source of ambiguity we don't want to fight initially.

### Step 6: Initialization strategy (no COLMAP, no learned prior)

This is the hardest part. We have no priors. The plan:

1. **Gaussians**: random positions in a ball of radius 1, random colors uniform in [0,1], small isotropic scale (~0.02), opacity 0.1. Start with **~100k splats**, not 10k — under cold start we need enough degrees of freedom that some random splats happen to land near the true geometry. Ten thousand is too few to cover a real scene without depth priors; the optimizer can't densify into structure that no splat is near.

2. **Cameras**: the right init depends on the capture pattern, which we won't know automatically. Strategies to support and pick between:
   - **Object-centric orbit** (e.g. walking around a statue): hemisphere init at radius 3 looking at origin, sequential along the path (each frame near the previous one's pose).
   - **Forward-facing pan**: cameras roughly co-located, looking outward in slowly-rotating directions.
   - **Translational sweep**: cameras along a line, parallel orientations.
   - **Default for the LocalSend test videos**: assume sequential capture, init each camera near the previous one with small random perturbation. Initial absolute placement: hemisphere arc.

3. **Coarse-to-fine schedule** (rough plan, to be tuned by experiment):
   - **Stage A (pose-dominant warm-up)** — iters 0 to ~3k: 200×200 resolution, ~50k splats, high camera LR, gaussian LRs reduced 10×, no densification. Goal: poses reach a coarse-correct arrangement. Loss should drop sharply if cold start is going to work at all.
   - **Stage B (joint refinement)** — iters 3k to ~15k: 400×400, full splat count, full LRs, gradual camera LR decay. Densification gated on pose stability.
   - **Stage C (detail)** — iters 15k onward: 800×800 or full resolution, low camera LR, densification active, opacity resets.

4. **Run multiple inits in parallel** (when possible). Cold start is non-convex; the cheapest way to deal with that is to run 4–8 different random seeds and keep the best one. Each costs $0.06 on a 4090.

5. **Failure modes to instrument**:
   - All cameras collapse to the same pose (degenerate solution where every view looks identical).
   - Splats collapse to a single point.
   - Loss plateaus high with poses still moving (under-parameterized).
   - Loss drops fast then explodes when densification kicks in (premature densification on bad geometry).

   Log per-camera `||ξ||` over time, per-splat opacity histogram, rendered-vs-target deltas. We need to *see* failures to fix them.

### Step 7: Export and verification

- **.ply schema**: write the GraphDECO 3DGS convention so MetalSplatter and other viewers can read it. Fields: `x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 f_rest_0..f_rest_44 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`. Conventions: opacity stored in **logit space** (apply `sigmoid` at render), scales stored in **log space** (apply `exp` at render), rotation as quaternion in **wxyz** order, normals can be zeroed if not used. Add a round-trip test that exports a small splat set and re-renders it through MetalSplatter (or a Python .ply reader) before building the full Mac viewer.
- Export recovered camera poses (JSON: per-camera `{quat: [w,x,y,z], trans: [x,y,z], fx, fy, cx, cy, width, height}`).
- Test with a known synthetic scene (see verification section).

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

In strict order:

1. **Camera gradient correctness** (M1): synthetic scene with known splats and known camera poses. Perturb a camera by a small `ξ`, render, compute loss against the ground-truth render. Confirm that:
   - `loss.backward()` populates `cameras.xi.grad`
   - finite-difference gradient (`(L(ξ+ε) - L(ξ-ε)) / 2ε`) matches autodiff to ~3 decimal places
   - a single Adam step from a perturbed pose moves loss downward
2. **Easy cold start — synthetic** (M2): textured cube scene. Initialize splats randomly and cameras at slightly-wrong poses (small noise). Verify both converge. Then increase pose noise progressively to find the cold-start radius of convergence.
3. **Hard cold start — synthetic** (M2): same scene, but cameras initialized fully randomly on the hemisphere. This is the canary for "does cold start work at all?"
4. **Easy real video** (M3): pick the easiest LocalSend clip (`VID_20260503_125941184.mp4`, 24s) — likely object-centric orbit. Train and inspect.
5. **Hard real video** (M4): the harder clips. Honest acceptance criterion: produces a viewable .ply where the recovered cameras roughly trace the actual capture path. Not "matches state-of-the-art reconstruction quality."

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

- **M0: Apple Silicon benchmark** — install gsplat, measure iter/sec at 100k splats × {200², 400², 800²} on this Mac. Decide: local-only, hybrid, or RunPod-only inner dev loop.
- **M1: Camera gradient correctness** — verify gsplat's `viewmats` autograd matches finite-difference. Switch `camera_model.py` from the wrapper to direct `viewmats` + SE(3) tangent parameterization.
- **M2: Cold start on synthetic data** — known textured scene, random init, verify both poses and splats converge. Map the radius of convergence.
- **M3: Cold start on the easiest LocalSend video** — produces viewable .ply with cameras tracing the capture path.
- **M4: Cold start on the harder LocalSend videos** — may require multi-seed runs, schedule changes, or just fail informatively.
- **M5: RunPod deployment** — Dockerized API for offloading runs that are too slow locally or for parallel multi-seed sweeps.
- **M6: Mac viewer** — .ply round-trip with MetalSplatter, orbit camera, recovered-cameras-as-frustums overlay.
- **M7: Android app** — capture, upload, download .ply, render locally.

## Immediate Actions

- [x] Create GitHub repo `gaussian_splat`
- [x] Commit PLAN.md + initial project structure
- [ ] **M0 — benchmark gsplat on Apple Silicon** (do this first, it gates everything else)
- [ ] Replace `transform_to_camera` wrapper with direct `viewmats` + SE(3) tangent parameterization in `training/camera_model.py`
- [ ] Implement camera gradient correctness test (M1)
- [ ] Build synthetic cube scene + cold-start convergence test (M2)
- [ ] Cold start on the easiest LocalSend video (M3)
- [ ] Cold start on the harder LocalSend videos (M4)
- [ ] Deploy to RunPod (M5)
- [ ] Build Mac viewer (M6)
- [ ] Build Android app (M7)
