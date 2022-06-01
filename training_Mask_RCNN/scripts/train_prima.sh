#!/bin/bash

python train_net.py \
    --dataset_name          prima-layout \
    --json_annotation_train ../data/prima/annotations-train.json \
    --image_path_train      ../data/prima/Images \
    --json_annotation_val   ../data/prima/annotations-val.json \
    --image_path_val        ../data/prima/Images \
    --config-file           ../configs/prima/mask_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/prima/mask_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 