#!/bin/bash

CHECKPOINT_FILE=<full path to the checkpoint to evaluate>
TEST_IMAGES_DIR=<full path to the root directory of the test images>
OUTPUT_DIR=<full path to the output directory to save all test related results>
TEST_JSON_FILE=<full path to the MS COCO test JSON file>
RESULT_COCO_FILE=<full path to the MS COCO format result JSON file. This file will be generated after the evaluation>
torchrun --nproc_per_node=2 evaluate.py        \
                          --checkpoint ${CHECKPOINT_FILE} \
                          --world-size  2   \
                          --workers 8   \
                          --aspect-ratio-group-factor 3 \
                          --dataset coco    \
                          -b 8    \
                          --model fasterrcnn_resnet50_fpn_v2   \
                          --trainable-backbone-layers 5     \
                          --test-images-dir ${TEST_IMAGES_DIR}  \
                          --test-json-file ${TEST_JSON_FILE} \
                          --output-dir ${OUTPUT_DIR}    \
                          --result-coco-file ${RESULT_COCO_FILE
