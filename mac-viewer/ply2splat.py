#!/usr/bin/env python3
"""Convert 3DGS PLY to .splat format for web viewers.

.splat format: per-gaussian, tightly packed:
  position: 3x float32 (12 bytes)
  scale: 3x float32 (12 bytes)  -- log space
  color: 4x uint8 (4 bytes) -- RGBA, 0-255
  rotation: 4x uint8 (4 bytes) -- quaternion, normalized to [0,255]
Total: 32 bytes per gaussian
"""
import struct
import sys
import math
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ply2splat.py input.ply [output.splat]")
        sys.exit(1)

    ply_path = sys.argv[1]
    splat_path = sys.argv[2] if len(sys.argv) > 2 else ply_path.rsplit('.', 1)[0] + '.splat'

    with open(ply_path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if line.strip() == b'end_header':
                break

        header_text = header.decode()
        num_verts = int([l for l in header_text.split('\n') if 'element vertex' in l][0].split()[-1])

        # Parse properties
        props = []
        in_vertex = False
        for line in header_text.split('\n'):
            if line.startswith('element vertex'):
                in_vertex = True
            elif line.startswith('element'):
                in_vertex = False
            elif in_vertex and line.startswith('property float'):
                props.append(line.split()[2])

        prop_idx = {name: i for i, name in enumerate(props)}
        stride = len(props) * 4

        # Read all binary data
        data = np.frombuffer(f.read(num_verts * stride), dtype='<f4').reshape(num_verts, len(props))

    print(f"Read {num_verts} gaussians with {len(props)} properties")

    # Extract fields
    positions = data[:, [prop_idx['x'], prop_idx['y'], prop_idx['z']]]

    # Scales (stored as log in PLY)
    scales = np.exp(data[:, [prop_idx['scale_0'], prop_idx['scale_1'], prop_idx['scale_2']]])

    # SH DC -> RGB
    SH_C0 = 0.28209479177387814
    dc0 = data[:, prop_idx['f_dc_0']]
    dc1 = data[:, prop_idx['f_dc_1']]
    dc2 = data[:, prop_idx['f_dc_2']]
    r = np.clip((0.5 + SH_C0 * dc0) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + SH_C0 * dc1) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + SH_C0 * dc2) * 255, 0, 255).astype(np.uint8)

    # Opacity (stored as logit in PLY)
    opacity = sigmoid(data[:, prop_idx['opacity']])
    a = np.clip(opacity * 255, 0, 255).astype(np.uint8)

    # Rotation quaternion (wxyz in PLY)
    rot = data[:, [prop_idx['rot_0'], prop_idx['rot_1'], prop_idx['rot_2'], prop_idx['rot_3']]]
    # Normalize
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = rot / np.maximum(rot_norm, 1e-8)
    # Convert to uint8: map [-1, 1] -> [0, 255]
    rot_u8 = np.clip((rot * 128 + 128), 0, 255).astype(np.uint8)

    # Write .splat file
    with open(splat_path, 'wb') as f:
        for i in range(num_verts):
            # Position (3x float32)
            f.write(struct.pack('<fff', *positions[i]))
            # Scale (3x float32, log space)
            f.write(struct.pack('<fff', *np.log(scales[i])))
            # Color RGBA (4x uint8)
            f.write(struct.pack('BBBB', r[i], g[i], b[i], a[i]))
            # Rotation (4x uint8)
            f.write(struct.pack('BBBB', *rot_u8[i]))

    print(f"Wrote {splat_path} ({num_verts * 32} bytes)")


if __name__ == '__main__':
    main()
