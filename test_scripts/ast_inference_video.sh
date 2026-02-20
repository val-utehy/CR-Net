#!/bin/bash
set -e

EXPERIMENT_NAME='ast_rafael_v2_sharpening'
WHICH_EPOCH=8
GPU_ID=0
INPUT_VIDEO='/home/share/RAFAEL_7_Dark/HangzhouDemo.mp4'
OUTPUT_VIDEO='/workspace/RAFAEL_7_Dark/output_vid/output_full_resolution.mp4'
CHECKPOINTS_DIR='./checkpoints'
CYCLE_DURATION=20
MAX_HEIGHT=720
# ------------------------------------

python inference_video.py \
    --name "${EXPERIMENT_NAME}" \
    --which_epoch "${WHICH_EPOCH}" \
    --gpu_ids "${GPU_ID}" \
    --video_path "${INPUT_VIDEO}" \
    --max_resolution_height ${MAX_HEIGHT} \
    --degree_step 1 \
    # --no_audio

echo "Video inference finished."
