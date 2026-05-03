"""Launch a training run on RunPod.

Creates a GPU pod, uploads the video, runs training, downloads results.

Usage:
    python3 run_training.py --video /path/to/video.mp4
    python3 run_training.py --video /path/to/video.mp4 --gpu RTX4090
"""

import argparse
import os
import sys
import time
import runpod


def main():
    parser = argparse.ArgumentParser(description="Launch gaussian splat training on RunPod")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="GPU type")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to extract")
    parser.add_argument("--num-gaussians", type=int, default=100000, help="Initial gaussian count")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set RUNPOD_API_KEY env var or pass --api-key")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    runpod.api_key = args.api_key

    video_name = os.path.basename(args.video)

    # Create pod with training setup
    print(f"Creating RunPod pod with {args.gpu}...")

    # Use a PyTorch CUDA image and install deps on startup
    pod = runpod.create_pod(
        name=f"gsplat-train-{int(time.time())}",
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        cloud_type="SECURE",
        volume_in_gb=20,
        container_disk_in_gb=20,
        ports="22/tcp",
        docker_args="",
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print("Waiting for pod to start...")

    # Wait for pod to be ready
    while True:
        status = runpod.get_pod(pod_id)
        pod_status = status.get("desiredStatus", "")
        runtime = status.get("runtime", {})
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            break
        print(f"  Status: {pod_status}...")
        time.sleep(10)

    print(f"Pod running!")

    # Get SSH info
    runtime = status["runtime"]
    ports = runtime.get("ports", [])
    ssh_port = None
    ssh_host = None
    for p in ports:
        if p.get("privatePort") == 22:
            ssh_host = p.get("ip")
            ssh_port = p.get("publicPort")
            break

    if not ssh_host:
        print("WARNING: Could not find SSH port. You may need to connect manually.")
        print(f"Pod ID: {pod_id}")
        print("Use RunPod web terminal to run the following commands:")
    else:
        print(f"SSH: ssh root@{ssh_host} -p {ssh_port}")

    # Print manual instructions (most reliable approach)
    print(f"""
=== Training Instructions ===

1. Connect to the pod:
   runpodctl exec {pod_id} bash

2. Set up the environment:
   pip install gsplat numpy opencv-python Pillow tqdm plyfile pytorch-msssim

3. Clone the repo:
   git clone https://github.com/simon-liesinger/gaussian_splat.git
   cd gaussian_splat/training

4. Upload video (from your local machine):
   runpodctl send {args.video}
   # Or use the RunPod web UI to upload

5. Run training:
   python3 train.py --input {video_name} --output /workspace/output --max-frames {args.max_frames} --num-gaussians {args.num_gaussians} --save-interval 200

6. Download results:
   runpodctl receive /workspace/output/model.ply
   runpodctl receive /workspace/output/cameras.json
   runpodctl receive /workspace/output/snapshots/

7. Terminate pod:
   runpodctl remove pod {pod_id}

Pod ID: {pod_id}
""")


if __name__ == "__main__":
    main()
