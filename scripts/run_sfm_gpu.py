#!/usr/bin/env python3
"""Run SfM on a RunPod GPU for fast COLMAP processing.

CUDA-accelerated SIFT extraction + matching is 10-50x faster than CPU.
Uploads images, runs SfM on GPU, downloads sfm_result.json.

Usage:
    RUNPOD_API_KEY=... python3 scripts/run_sfm_gpu.py --images path/to/images/
    RUNPOD_API_KEY=... python3 scripts/run_sfm_gpu.py --video path/to/video.mp4 --max-frames 100
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

RUNPOD_BASE = "https://rest.runpod.io/v1"


def get_ssh_pubkey():
    for path in [
        os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go.pub"),
        os.path.expanduser("~/.ssh/id_ed25519.pub"),
        os.path.expanduser("~/.ssh/id_rsa.pub"),
    ]:
        if os.path.exists(path):
            return open(path).read().strip(), path
    return None, None


def api_request(method, path, api_key, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{RUNPOD_BASE}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"API error {e.code}: {e.read().decode()}")


def main():
    parser = argparse.ArgumentParser(description="Run SfM on RunPod GPU")
    parser.add_argument("--images", help="Directory of images")
    parser.add_argument("--video", help="Video file (will extract frames)")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames from video")
    parser.add_argument("--output", default="sfm_result.json", help="Output JSON path")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Set RUNPOD_API_KEY or pass --api-key")
    if not args.images and not args.video:
        sys.exit("Pass --images or --video")

    pubkey, pubkey_path = get_ssh_pubkey()
    if not pubkey:
        sys.exit("No SSH public key found")
    ssh_key = pubkey_path.replace(".pub", "")

    # Create tarball of images
    print("Packaging images...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = tmp.name

    if args.video:
        # Extract frames first
        import cv2
        tmpdir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // args.max_frames)
        count = 0
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(tmpdir, f"frame_{count:04d}.jpg"), frame)
                count += 1
        cap.release()
        print(f"  Extracted {count} frames")
        image_dir = tmpdir
    else:
        image_dir = args.images

    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._'):
                tar.add(os.path.join(image_dir, f), arcname=f)

    tar_size = os.path.getsize(tar_path) / 1e6
    print(f"  Tarball: {tar_size:.1f} MB")

    # Bootstrap: install pycolmap-cuda12, run SfM, save result
    bootstrap = r"""
set -u
mkdir -p /workspace
exec > >(tee -a /workspace/bootstrap.log) 2>&1
setup_ssh() { ssh-keygen -A 2>&1||true; mkdir -p /root/.ssh /run/sshd; chmod 700 /root/.ssh; :>/root/.ssh/authorized_keys; for k in "${PUBLIC_KEY:-}" "${USER_PUBLIC_KEY:-}"; do [ -n "$k" ]&&printf '%s\n' "$k">>/root/.ssh/authorized_keys; done; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -e; }
setup_ssh||true
(
    pip install --no-cache-dir --break-system-packages pycolmap numpy scipy 2>&1|tail -3
    while [ ! -f /workspace/images.tar.gz ]; do sleep 2; done
    mkdir -p /workspace/images && cd /workspace/images && tar xzf /workspace/images.tar.gz
    echo "$(ls *.jpg *.jpeg *.png 2>/dev/null | wc -l) images"

    python3 -c "
import pycolmap, os, json, tempfile, shutil, numpy as np

image_dir = '/workspace/images'
work_dir = tempfile.mkdtemp(prefix='colmap_')
db_path = os.path.join(work_dir, 'database.db')

print('Extracting features (GPU)...')
pycolmap.extract_features(db_path, image_dir)

num_images = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
if num_images > 50:
    print(f'Sequential matching ({num_images} images)...')
    pycolmap.match_sequential(db_path)
else:
    print(f'Exhaustive matching ({num_images} images)...')
    pycolmap.match_exhaustive(db_path)

print('Incremental mapping...')
sparse_dir = os.path.join(work_dir, 'sparse')
os.makedirs(sparse_dir)
maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)

if not maps:
    print('SfM FAILED')
    exit(1)

rec = maps[0]
print(f'Reconstruction: {rec.num_images()} images, {rec.num_points3D()} points')

cameras_out, intrinsics_out, image_names = [], [], []
for image in sorted(rec.images.values(), key=lambda i: i.name):
    cfw = image.cam_from_world()
    R = np.array(cfw.rotation.matrix())
    t = np.array(cfw.translation)
    vm = np.eye(4, dtype=np.float32)
    vm[:3,:3] = R; vm[:3,3] = t
    cameras_out.append(vm.tolist())
    cam = rec.cameras[image.camera_id]
    params = cam.params
    if cam.model_name in ('SIMPLE_PINHOLE','SIMPLE_RADIAL'):
        fx=fy=params[0]; cx,cy=params[1],params[2]
    elif cam.model_name in ('PINHOLE','OPENCV'):
        fx,fy=params[0],params[1]; cx,cy=params[2],params[3]
    else:
        fx=fy=params[0]; cx=cam.width/2; cy=cam.height/2
    intrinsics_out.append([fx,fy,cx,cy])
    image_names.append(image.name)

points, colors = [], []
for p in rec.points3D.values():
    points.append(p.xyz.tolist())
    colors.append(p.color.tolist())

result = {
    'cameras': cameras_out, 'intrinsics': intrinsics_out,
    'image_names': image_names,
    'points3d': points, 'point_colors': colors,
}
with open('/workspace/sfm_result.json', 'w') as f:
    json.dump(result, f)
print(f'Saved: {len(cameras_out)} cameras, {len(points)} points')
shutil.rmtree(work_dir, ignore_errors=True)
"
    echo "SFM_DONE"
)&
exec sleep infinity
"""

    # Launch pod
    print("Launching RunPod GPU pod...")
    body = {
        "name": "sfm-gpu",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "computeType": "GPU",
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A6000"],
        "gpuCount": 1, "containerDiskInGb": 20, "volumeInGb": 5, "volumeMountPath": "/workspace",
        "dockerStartCmd": ["bash", "-lc", bootstrap],
        "env": {"USER_PUBLIC_KEY": pubkey},
        "ports": ["22/tcp"],
        "countryCodes": ["US"],
        "minDownloadMbps": 500,
    }
    result = api_request("POST", "/pods", args.api_key, body)
    pod_id = result["id"]
    print(f"  Pod: {pod_id}")

    # Wait for SSH
    ip = ssh_port = None
    for _ in range(60):
        pod = api_request("GET", f"/pods/{pod_id}", args.api_key)
        ip = pod.get("publicIp")
        ssh_port = (pod.get("portMappings") or {}).get("22")
        if ip and ssh_port:
            break
        time.sleep(5)

    if not ip or not ssh_port:
        sys.exit("Pod failed to start")

    print(f"  SSH: {ip}:{ssh_port}")
    time.sleep(5)

    # Upload images
    print("Uploading images...")
    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    subprocess.run(["scp", "-P", str(ssh_port)] + ssh_opts + [tar_path, f"root@{ip}:/workspace/uploading.tar.gz"],
                   capture_output=True, timeout=600)
    subprocess.run(["ssh", "-p", str(ssh_port)] + ssh_opts + [f"root@{ip}",
                   "mv /workspace/uploading.tar.gz /workspace/images.tar.gz"],
                   capture_output=True, timeout=30)
    print("  Uploaded")

    # Wait for SfM to complete
    print("Waiting for SfM...")
    for attempt in range(120):
        try:
            result = subprocess.run(
                ["ssh", "-p", str(ssh_port)] + ssh_opts + [f"root@{ip}",
                 "grep SFM_DONE /workspace/bootstrap.log 2>/dev/null"],
                capture_output=True, text=True, timeout=10)
            if "SFM_DONE" in result.stdout:
                break
        except:
            pass
        time.sleep(5)
    else:
        print("Timed out waiting for SfM")
        api_request("POST", f"/pods/{pod_id}/stop", args.api_key)
        sys.exit(1)

    # Download result
    print("Downloading result...")
    subprocess.run(["scp", "-P", str(ssh_port)] + ssh_opts +
                   [f"root@{ip}:/workspace/sfm_result.json", args.output],
                   capture_output=True, timeout=60)

    # Stop pod
    api_request("POST", f"/pods/{pod_id}/stop", args.api_key)
    print(f"Pod stopped. Result saved to {args.output}")

    # Show summary
    with open(args.output) as f:
        d = json.load(f)
    print(f"  {len(d['cameras'])} cameras, {len(d['points3d'])} points")

    # Cleanup
    os.unlink(tar_path)


if __name__ == "__main__":
    main()
