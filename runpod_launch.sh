#!/usr/bin/env bash
set -euo pipefail

# 1) Environment prep
echo "=== Updating pip and installing system deps ==="
pip install --upgrade pip

# (Optional) If you need git-lfs for large model files:
# apt-get update && apt-get install -y git-lfs && git-lfs install

# 2) Clone your repo (or pull latest)
WORKDIR=/workspace
REPO_URL="https://github.com/temefford/fluxft.git"
BRANCH="main"

echo "=== Cloning repo ==="
cd $WORKDIR
if [ -d fluxft ]; then
  echo "fluxft directory exists, pulling latest..."
  cd fluxft && git fetch origin && git checkout $BRANCH && git pull
else
  git clone --branch $BRANCH $REPO_URL
  cd fluxft
fi

# 3) Install the package
echo "=== Installing Python dependencies ==="
pip install -e .

# 4) (Optional) Verify CUDA & GPU
echo "=== GPU status ==="
nvidia-smi

# 5) Run the fine-tune
CONFIG_FILE=config.yaml   # adjust if you named it differently

echo "=== Starting fine-tuning with config '${CONFIG_FILE}' ==="
# Log everything to a file
fluxft finetune --cfg-path "${CONFIG_FILE}" 2>&1 | tee training.log
TRAIN_EXIT=${PIPESTATUS[0]}

# 6) Shutdown on success
if [ "$TRAIN_EXIT" -eq 0 ]; then
  echo "=== Training complete, shutting down ==="
  # give logs a moment to flush
  sync
  sleep 5
  sudo shutdown -h now
else
  echo "=== Training FAILED (exit code $TRAIN_EXIT) ==="
  echo "Machine will remain up for debugging."
  exit $TRAIN_EXIT
fi