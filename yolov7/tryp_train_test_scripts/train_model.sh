#!/usr/bin/env bash

python -m torch.distributed.launch  \
    --nproc_per_node 2  \
    --master_port 9527  \
    train.py    \
        --workers 8     \
        --device 0,1    \
        --sync-bn   \
        --batch-size 32    \
        --data data/coco_tryp.yaml   \
        --img 640 640   \
        --cfg cfg/training/yolov7_tryp.yaml  \
        --weights yolov7.pt    \
        --name yolov7_default_march_10th_test_nms_0.4   \
        --hyp data/hyp.scratch.custom.yaml

