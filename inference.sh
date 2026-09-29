#!/bin/bash

gpu_ids=0

dataset_dir=./dataset/LibriSpeech

checkpoint="./exp/Bias/01/last.ckpt"

export CUDA_VISIBLE_DEVICES=$gpu_ids
python ./recipes/inference.py \
    --eval-dataset "$dataset_dir/test-clean.json" \
    --checkpoint "$checkpoint" \
    --batch-size 200 \
    --load-encoder \
    --run-asr \
    --run-bias \

python ./scripts/evaluation.py
