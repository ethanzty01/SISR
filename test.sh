SISR python test.py \
  --checkpoint "./checkpoints/DPSR_x4_0917_2222.pth" \
  --scale 4 \
  --channel_nums 32 \
  --subnet-channels 16 \
  --num_blocks 5 \
  --in_channels 3 \
  --device cuda
