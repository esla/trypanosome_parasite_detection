#!/usr/bin/env bash

MODEL_WEIGHTS=/home/esla/server_drive/esla/research/experiments_for_papers/trypa-detection/yolo_station/yolov7_original/yolov7/runs/train/yolov7_default_march_10th_test_nms_0.4_conf_0.001/weights/best_297.pt
PARTITION=test
python test.py      \
            --data data/coco_tryp.yaml       \
            --img 640       \
            --batch 32      \
            --conf 0.001    \
            --iou 0.3  \
            --task ${PARTITION}  \
            --save-json   \
            --gt-jsonfile   \
            --dest-jsonfile     \
            --save-txt \
            --project /home/esla/server_drive/esla/research/experiments_for_papers/trypa-detection/yolo_station/yolov7_original/evaluations/${PARTITION}     \
            --device 0  \
            --weights ${MODEL_WEIGHTS}     \
            --name yolov7_default_march_10th_test_nms_0.3_conf_0.001

