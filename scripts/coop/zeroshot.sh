#!/bin/bash

# custom config
DATA=/media/zyx/OOD_Related/datasets/
TRAINER=ZeroshotCLIP
DATASET=imagenet
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
sub=new # all, base or new

python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DATASET}/${sub} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES ${sub}