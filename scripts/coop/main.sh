#!/bin/bash
# rm -r outputfmxtext1-num0-match-yz/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1
# rm -r outputIMAGENET10-0.89-0.1
# rm -r outputIMAGENET1k2025-0.98-0.05/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1

# custom config
DATA=/media/zyx/OOD_Related/datasets/
TRAINER=CoOp

DATASET=imagenet20
CFG=vit_b16_ep50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
sub=all


for SEED in 1
do
    DIR=outputIMAGENET20-0.9-0.05/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=2 python /media/chaod/fmx_newcoop/CoOp-main/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file /media/chaod/fmx_newcoop/CoOp-main/configs/datasets/${DATASET}.yaml \
        --config-file /media/chaod/fmx_newcoop/CoOp-main/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}\
        DATASET.SUBSAMPLE_CLASSES ${sub}
    fi
done