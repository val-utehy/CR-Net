#!/bin/bash
set -e

EXPERIMENT_NAME='ast_rafael_v2_sharpening'
IMAGE_FOLDER='/home/share/RAFAEL_8/VOC_2012_dark_YOLO_format/images/trainval'
WHICH_EPOCH='latest'
MAX_IMG_HEIGHT=960
GPU_ID="0"

#@HOURS="16.0"
HOURS="16.0 17.0 18.0 19.0 20.0 21.0 22.0"
#HOURS="17.0 17.5 18.0 18.5 19.0 19.5 20.0"

ADD_MARK=false

RESULTS_DIR='./results_v2'
CHECKPOINTS_DIR='./checkpoints_v2'
TIMESTAMP_FLAG=""
if [ "$ADD_MARK" = true ]; then
    TIMESTAMP_FLAG="--add_timestamp"
fi
python test.py \
    --name "${EXPERIMENT_NAME}" \
    --image_dir "${IMAGE_FOLDER}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --which_epoch "${WHICH_EPOCH}" \
    --gpu_ids "${GPU_ID}" \
    --hours ${HOURS} \
    --load_from_opt_file \
    --max_height "${MAX_IMG_HEIGHT}" \
    ${TIMESTAMP_FLAG}

echo "Testing finished. Results are in ${RESULTS_DIR}/${EXPERIMENT_NAME}/"
