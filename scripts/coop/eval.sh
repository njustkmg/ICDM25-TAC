#!/bin/bash

# custom config
DATA=/media/zyx/OOD_Related/datasets/
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=imagenet
CFG=vit_b16_ep50
sub=all

# for sub in 'base' 'new' 'all'; do
for SEED in 1
do
    CUDA_VISIBLE_DEVICES=2 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/otest2/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${sub} \
    --model-dir /media/chaod/fmx_newcoop/CoOp-main/output-zyxbase-1.0/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1 \
    --load-epoch ${SEED} \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.SUBSAMPLE_CLASSES ${sub}

done