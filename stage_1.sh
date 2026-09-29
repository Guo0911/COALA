#!/bin/bash

gpu_ids=0

bias_mode=train

dataset_dir=./dataset
dataset_name=LibriSpeech
train_dataset=train-960h
valid_dataset=dev-clean

adapter_model=conv-branchformer
connector_model=cnn
backbone_model=HuggingFaceTB/SmolLM2-135M-Instruct

wandb_project="COALA"
export CUDA_VISIBLE_DEVICES=$gpu_ids
python ./recipes/train_ASR.py \
    --train-dataset "$dataset_dir/$dataset_name/$train_dataset.json" \
    --valid-dataset "$dataset_dir/$dataset_name/$valid_dataset.json" \
    --adapter-model "$adapter_model" \
    --connector-model "$connector_model" \
    --backbone-model "$backbone_model" \
    --feature-dim 1280 \
    --connector-dim 576 \
    --ctc-gate-dim 128 \
    --connector-k 4 \
    --ctc-weight 0.5\
    --use-lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --max-lr "1e-4" \
    --total-training-step 700000 \
    --batch-size 4 \
    --train-batch-per-epoch 70000 \
    --grad-accumulate-steps 4 \
    --num-device 1 \
    --bias-mode "$bias_mode" \
    --wandb "$wandb_project"
