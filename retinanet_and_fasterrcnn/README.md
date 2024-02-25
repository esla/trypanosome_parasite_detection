# Trypanosome detection with Faster R-CNN and RetinaNet

This folder contains training codes and scripts for trypanosome parasite detection. However,
they are general enough for any object detection task that is based on MS COCO JSON format.
The purpose of this repo is to serve as a reference implementation of training the Faster R-CNN
and RetinaNet models for trypanosome parasite detection on the Tryp dataset.

### Requirements
We included the versions
of the packages that we have tested on in the [requirements.txt](requirements.txt) file. If you 
have issues running the code on the latest versions of the required packages, ensure that you install
the versions in the requirements.txt file.

All models were trained on two Titan RTX GPUs. 

## Training
We provide the training and evaluation scripts in the [scripts](scripts) folder. You have to modify the
required paths in the scripts to match your settings. The content of the scripts are:

### Dataset preparation
Download the Tryp dataset into your preferred locations. The paths to the dataset information
are needed in the training scripts. The dataset annotation information is in the MS COCO JSON format.
We have used the trypanosome dataset hosted at [Figshare](https://doi.org/10.6084/m9.figshare.22825787.v1)
to train Faster-RCNN and RetinaNet using this repository.

### Faster R-CNN ResNet-50 FPN 
An example script is [here](scripts/train_faster_rcnn_resnet50_fpn.sh).
```
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
```



### RetinaNet ResNet-50 FPN 
An example script is [here](scripts/train_retinanet.sh).
```
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
                          --model retinanet_resnet50_fpn_v2   \
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
```

## Evaluation
### Evaluating a pre-trained Faster R-CNN ResNet-50 FPN model
```
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
                          --result-coco-file ${RESULT_COCO_FILE}
```
To evaluate a pre-trained RetinaNet model, simple change the model name to `retinanet_resnet50_fpn_v2` and
the CHECKPOINT_FILE to the RetinaNet checkpoint.

## Results
All results from the training or evaluation can be found in the `OUTPUT_DIR` that you specified in the
training or evaluation scripts. The results include: the saved checkpoints and the tensorboard logs.
