#!/bin/bash

# This script is used to prepare the environment for training and inference of the YOLOv7 model on the Tryp dataset.

# Move config and script files to their respective directories in the YOLOv7 directory.
cp -v ./tryp_train_test_scripts/* ./yolov7/scripts/
cp -v ./tryp_config_yamls/cfg/training/yolov7_tryp.yaml ./yolov7/cfg/training/
cp -v ./tryp_config_yamls/data/coco_tryp.yaml ./yolov7/data/

# Update the coco_tryp.yaml file with the correct paths to the Tryp dataset.
# (a) Check if the Tryp dataset root directory is provided as an argument
if [ -z "$1" ]; then
  # Download the Tryp dataset from the following link:
  # Variable not provided, update it with a default value
    dataset_root_dir=$(pwd)/Tryp
else
    # Variable provided as an argument, update it with the provided value
    dataset_root_dir="$1"
fi
# (b) Update the coco_tryp.yaml file with the correct paths to the Tryp dataset.
sed "/s/ROOTDIR/${dataset_root_dir}/g" ./yolov7/data/coco_tryp.yaml

# clone the YOLOv7 repository
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
