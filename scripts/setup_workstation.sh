#!/usr/bin/env bash
# Create the training virtual environment and install dependencies.
#
# The only non-obvious step is the pybullet build on macOS.  pybullet ships no
# Apple-Silicon wheel, so pip compiles it from source; the bundled copy of zlib
# defines `fdopen(fd,mode) NULL` on Darwin, which then collides with the real
# declaration in the macOS SDK's <stdio.h> and aborts the build.  Predefining
# `fdopen` satisfies zlib's `#ifndef` guard so it leaves the system header alone.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
echo "==> Creating virtual environment with $($PYTHON -V)"
"$PYTHON" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip setuptools wheel

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> macOS detected: applying zlib/SDK fdopen workaround for the pybullet build"
  export CFLAGS="${CFLAGS:-} -Dfdopen=fdopen"
  export CPPFLAGS="${CPPFLAGS:-} -Dfdopen=fdopen"
fi

echo "==> Installing requirements (pybullet compiles from source; expect a few minutes)"
./.venv/bin/pip install -r requirements.txt

echo "==> Verifying the installation"
./.venv/bin/python -c "
import gymnasium as gym, panda_gym, stable_baselines3
env = gym.make('PandaReach-v3'); obs, _ = env.reset(seed=0)
assert set(obs) == {'observation', 'achieved_goal', 'desired_goal'}
print('  panda-gym  ', obs['observation'].shape, env.action_space)
print('  sb3        ', stable_baselines3.__version__)
env.close()
"
echo "==> Done.  Activate with:  source .venv/bin/activate"
