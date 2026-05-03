#!/bin/bash
# Launch the Gaussian Splat Viewer
# Usage: ./run.sh [path/to/model.ply]
cd "$(dirname "$0")"
swift run GaussianViewer "$@"
