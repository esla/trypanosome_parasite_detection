#!/usr/bin/env bash

TRAIN_IMAGES_DIR=<full path to the training root directory of the training images>
VAL_IMAGES_DIR=<full path to the root directory of the validation images>
OUTPUT_DIR=<full path to the output directory to save all training related results, such as checkpoints>
TENSORBOARD_LOG_DIR=<full path to the root directory to save tensorboard logs>
TRAIN_JSON_FILE=<full path to the MS COCO format train JSON file>
VAL_JSON_FILE=<full path to the MS COCO val JSON file>
torchrun --nproc_per_node=2 train.py        \
                          --world-size  2   \
                          --workers 8   \
                          --lr-scheduler cyclelr     \
                          --lr 0.005    \
                          --aspect-ratio-group-factor 3 \
                          --dataset coco    \
                          -b 8    \
                          --model fasterrcnn_resnet50_fpn_v2   \
                          --trainable-backbone-layers 5     \
                          --epochs 100   \
                          --weights FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT  \
                          --weights-backbone ResNet50_Weights.DEFAULT \
                          --train-images-dir ${TRAIN_IMAGES_DIR}    \
                          --val-images-dir ${VAL_IMAGES_DIR} \
                          --train-json-file ${TRAIN_JSON_FILE} \
                          --val-json-file ${VAL_JSON_FILE} \
                          --output-dir ${OUTPUT_DIR}    \
                          --tensorboard-rootdir ${TENSORBOARD_LOG_DIR}