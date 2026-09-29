#!/bin/bash

gpu_ids=0

dataset_dir=./dataset
dataset_name=LibriSpeech
train_dataset=train-960h
valid_dataset=dev-clean
bias_lists="./dataset/LibriSpeech/bias-lists/N=5000"

scoring_loss="DPD-loss"

checkpoint="./exp/ASR/01/last.ckpt"

wandb_project="COALA"
export CUDA_VISIBLE_DEVICES=$gpu_ids
python ./recipes/train_Bias.py \
    --train-dataset "$dataset_dir/$dataset_name/$train_dataset.json" \
    --valid-dataset "$dataset_dir/$dataset_name/$valid_dataset.json" \
    --bias-lists "$bias_lists" \
    --scoring-loss "$scoring_loss" \
    --checkpoint "$checkpoint" \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --max-lr "1e-4" \
    --total-training-step 490000 \
    --batch-size 1 \
    --train-batch-per-epoch 70000 \
    --grad-accumulate-steps 4 \
    --num-device 1 \
    --wandb "$wandb_project"
