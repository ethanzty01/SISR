#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n SISR python final_test.py \
  --threshold-sweep \
  --dpsr-checkpoint ./checkpoints/DPSR_x4_0815_1611.pth \
  --scale 4 --arm_patch_size 16 --arm_overlap 2 \
  --channel_nums 32 --subnet_channels 16 --num_blocks 5 \
  "$@"
