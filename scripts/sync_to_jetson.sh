#!/usr/bin/env bash
# Copy the robot-side code and the exported policies to the Jetson.
#
# Only what the robot needs is sent: the hardware package, the exported .npz
# policies and the workspace calibration.  The training code, checkpoints and
# replay buffers stay on the workstation — the Jetson never needs PyTorch.
set -euo pipefail

cd "$(dirname "$0")/.."

JETSON_HOST="${JETSON_HOST:-jetson@192.168.3.211}"
JETSON_DIR="${JETSON_DIR:-~/RL_project}"

echo "==> Syncing to ${JETSON_HOST}:${JETSON_DIR}"
ssh "$JETSON_HOST" "mkdir -p ${JETSON_DIR}/{hardware,policies,configs,results/hardware}"

rsync -az --delete hardware/ "${JETSON_HOST}:${JETSON_DIR}/hardware/"
rsync -az policies/ "${JETSON_HOST}:${JETSON_DIR}/policies/"
rsync -az configs/ "${JETSON_HOST}:${JETSON_DIR}/configs/"

echo "==> Verifying the robot-side environment"
ssh "$JETSON_HOST" "cd ${JETSON_DIR} && python3 -c \"
import sys; sys.path.insert(0, 'hardware')
import numpy, pymycobot
from numpy_policy import NumpyPolicy
from pathlib import Path
print('  numpy     ', numpy.__version__)
print('  pymycobot ', pymycobot.__version__)
for p in sorted(Path('policies').glob('*.npz')):
    print('  policy    ', p.name, NumpyPolicy.load(p))
\""

echo "==> Done."
echo "    Next:  ssh ${JETSON_HOST}"
echo "           cd ${JETSON_DIR} && python3 hardware/probe_arm.py"
