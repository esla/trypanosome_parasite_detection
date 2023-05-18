#!/usr/bin/env bash

WEIGHTS=/home/esla/server_drive/esla/research/experiments_for_papers/trypa-detection/yolo_station/yolov7_original/yolov7/runs/train/yolov7_default_march_10th_test_nms_0.4_conf_0.001/weights/best_297.pt
EXPERIMENT_NAME=yolov7_default_march_10th_test_nms_0.4_conf_0.001
SAVE_DIR=/home/esla/server_drive/esla/research/experiments_for_papers/trypa-detection/yolo_station/yolov7_original/evaluations/test

python detect.py    \
        --project journal_experiments  \
        --name  ${EXPERIMENT_NAME}    \
        --weights ${WEIGHTS}     \
        --conf 0.0     \
        --iou-thres    0.3  \
        --img-size 640  \
        --save-txt      \
        --save-conf     \
        --source ${SAVE_DIR}

