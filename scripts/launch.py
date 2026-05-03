#!/usr/bin/env python3
"""Launch a gaussian splat training run on RunPod.

Creates a GPU pod that clones the repo, installs deps, and waits for
a video to be uploaded via scp. Then runs training and saves results.

Usage:
    RUNPOD_API_KEY=... python3 scripts/launch.py --video ~/Downloads/VID_20260503_125941184.mp4
    RUNPOD_API_KEY=... python3 scripts/launch.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNPOD_BASE = "https://rest.runpod.io/v1"


def get_ssh_pubkey() -> str | None:
    for path in [
        os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go.pub"),
        os.path.expanduser("~/.ssh/id_ed25519.pub"),
        os.path.expanduser("~/.ssh/id_rsa.pub"),
    ]:
        if os.path.exists(path):
            return open(path).read().strip(), path
    return None, None


def bootstrap_cmd(save_interval: int, max_frames: int, num_gaussians: int) -> list[str]:
    """Produce a dockerStartCmd argv.

    Design follows kim-em/jung: sleep infinity as PID 1, sshd as daemon,
    training in background subshell. Nothing can kill the container.
    """
    script = rf"""
set -u
mkdir -p /workspace
exec > >(tee -a /workspace/bootstrap.log) 2>&1
echo "=== [bootstrap] $(date -u) starting ==="

# --- SSH ---
setup_ssh() {{
    ssh-keygen -A 2>&1 || true
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    : > /root/.ssh/authorized_keys
    for k in "${{PUBLIC_KEY:-}}" "${{USER_PUBLIC_KEY:-}}"; do
        [ -n "$k" ] && printf '%s\n' "$k" >> /root/.ssh/authorized_keys
    done
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd -e
    sleep 1
    ss -tlnp 2>/dev/null | grep -E ':22 ' || echo "[ssh] WARNING: :22 not listening"
}}
setup_ssh || echo "[ssh] setup_ssh failed but continuing"

# --- Training pipeline in background subshell ---
(
    echo "[bg] $(date -u) starting training pipeline"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[bg] no GPU"

    set -e
    cd /workspace
    git clone --depth 1 https://github.com/simon-liesinger/gaussian_splat.git gsplat_train
    cd gsplat_train/training

    pip install --no-cache-dir --break-system-packages \
        gsplat numpy opencv-python-headless Pillow tqdm plyfile pytorch-msssim 2>&1 | tail -5
    echo "[bg] $(date -u) deps installed"

    # Wait for video to appear (uploaded via scp)
    echo "[bg] Waiting for /workspace/input.mp4 ..."
    echo "[bg] Upload with: scp -P <port> -i ~/.runpod/ssh/RunPod-Key-Go VIDEO root@<ip>:/workspace/input.mp4"
    while [ ! -f /workspace/input.mp4 ]; do
        sleep 5
    done
    echo "[bg] $(date -u) video found, starting training"

    python3 train.py \
        --input /workspace/input.mp4 \
        --output /workspace/output \
        --max-frames {max_frames} \
        --num-gaussians {num_gaussians} \
        --save-interval {save_interval} \
        2>&1 | tee /workspace/train.log

    echo "[bg] $(date -u) training complete!"
    echo "[bg] Results in /workspace/output/"
    ls -la /workspace/output/
    ls -la /workspace/output/snapshots/ 2>/dev/null | head -20
) &

echo "[bootstrap] main process: sleep infinity"
exec sleep infinity
"""
    return ["bash", "-lc", script]


def post_pod(api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{RUNPOD_BASE}/pods",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API returned {e.code}: {e.read().decode()}")


def get_pod(api_key: str, pod_id: str) -> dict:
    req = urllib.request.Request(
        f"{RUNPOD_BASE}/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Launch gaussian splat training on RunPod")
    parser.add_argument("--video", help="Path to input video (uploaded via scp after pod starts)")
    parser.add_argument("--gpu", action="append", default=None,
                        help="GPU type (repeatable for fallback). Default: RTX 4090, A40, A100.")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--num-gaussians", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--country-codes", default="US")
    parser.add_argument("--min-download-mbps", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY must be set")

    pubkey, pubkey_path = get_ssh_pubkey()
    if not pubkey:
        sys.exit("No SSH public key found")

    gpus = args.gpu or ["NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A6000"]

    body = {
        "name": "gsplat-train",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "computeType": "GPU",
        "gpuTypeIds": gpus,
        "gpuCount": 1,
        "containerDiskInGb": 30,
        "volumeInGb": 10,
        "volumeMountPath": "/workspace",
        "dockerStartCmd": bootstrap_cmd(args.save_interval, args.max_frames, args.num_gaussians),
        "env": {
            "USER_PUBLIC_KEY": pubkey,
        },
        "ports": ["22/tcp"],
    }
    if args.country_codes.strip():
        body["countryCodes"] = [c.strip() for c in args.country_codes.split(",") if c.strip()]
    if args.min_download_mbps:
        body["minDownloadMbps"] = args.min_download_mbps

    if args.dry_run:
        print(json.dumps(body, indent=2))
        return

    print(f"Launching pod...")
    print(f"  GPU:     {gpus[0]} (fallbacks: {gpus[1:]})")
    print(f"  SSH key: {pubkey_path}")
    print()

    result = post_pod(api_key, body)
    pod_id = result.get("id") or result.get("podId") or "?"
    print(f"Pod created: {pod_id}")

    # Poll for SSH info
    print("Waiting for pod + SSH...")
    ssh_key_flag = f"-i {pubkey_path.replace('.pub', '')} " if pubkey_path and pubkey_path.endswith(".pub") else ""

    for attempt in range(60):
        try:
            pod = get_pod(api_key, pod_id)
        except Exception:
            time.sleep(10)
            continue

        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings") or {}
        ssh_port = port_mappings.get("22")

        if public_ip and ssh_port:
            print(f"\nPod ready!")
            print(f"  SSH:  ssh {ssh_key_flag}-p {ssh_port} root@{public_ip}")
            print(f"  Logs: ssh {ssh_key_flag}-p {ssh_port} root@{public_ip} 'tail -F /workspace/bootstrap.log /workspace/train.log 2>/dev/null'")

            if args.video and os.path.exists(args.video):
                print(f"\nWaiting 15s for sshd to start, then uploading video...")
                time.sleep(15)
                scp_cmd = [
                    "scp",
                    "-P", str(ssh_port),
                    "-i", pubkey_path.replace(".pub", ""),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    args.video,
                    f"root@{public_ip}:/workspace/input.mp4",
                ]
                print(f"  Running: {' '.join(scp_cmd)}")
                result = subprocess.run(scp_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("Video uploaded! Training will start automatically.")
                else:
                    print(f"SCP failed (may need to wait longer): {result.stderr.strip()}")
                    print(f"Retry manually:")
                    print(f"  scp -P {ssh_port} {ssh_key_flag}-o StrictHostKeyChecking=no {args.video} root@{public_ip}:/workspace/input.mp4")

            print(f"\nDownload results when done:")
            print(f"  scp -P {ssh_port} {ssh_key_flag}-r root@{public_ip}:/workspace/output/ ./output/")
            print(f"\nTerminate pod:")
            print(f"  curl -X POST -H 'Authorization: Bearer $RUNPOD_API_KEY' {RUNPOD_BASE}/pods/{pod_id}/stop")
            break

        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds", 0)
        print(f"  [{attempt+1}] status={status} uptime={uptime}s ip={public_ip} ssh={ssh_port}")
        time.sleep(10)
    else:
        print(f"\nTimed out. Check: https://www.runpod.io/console/pods/{pod_id}")


if __name__ == "__main__":
    main()
