#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n SISR python laplace_psnr.py \
  --model all \
  --scale 4 \
  --block_size 16 \
  --channel_nums 32 \
  --subnet_channels 16 \
  --num_blocks 5 \
  --in_channels 3 \
  --device cuda "$@"
