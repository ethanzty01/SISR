conda run --no-capture-output -n SISR python -u final_test.py \
  --model_name QDPSR \
  --dpsr-checkpoint "./checkpoints/QDPSR_x4_0917_2252.pth" \
  --scale 4 \
  --channel_nums 32 \
  --subnet_channels 16 \
  --num_blocks 5 \
  --in_channels 3 \
  --arm_threshold 20 \
  --arm_subnet_threshold 5 \
  --arm_patch_size 16 \
  --arm_overlap 2 \
  --device cuda
  # --model_name QDPSR \
  # --dpsr-checkpoint "./checkpoints/QDPSR_x4_0917_2252.pth" \
