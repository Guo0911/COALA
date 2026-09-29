import os
import time
import wandb
import torch
import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from models import asrLM as SpeechLLM
from dataset import Prompted_ASR_Dataset, ASR_Collator

def get_exp_path(save_path='./exp/ASR'):
    if os.path.exists(save_path):
        exist_ids = os.listdir(save_path)
    else:
        exist_ids = []

    current_ids = 1 # the first exp id is 01
    if exist_ids:
        if os.listdir(os.path.join(save_path, exist_ids[-1])): # if last id isn't empty
            current_ids += len(exist_ids)
        else:
            current_ids = exist_ids[-1]

    return os.path.join(save_path, str(current_ids).zfill(2))

def check_VRAM(required=20):
    required_memory = required * 1024**3

    print(f"Checking GPU memory on device {torch.cuda.current_device()}...")

    last_free_gb = 0

    check_times = 3
    while True:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3

        if (free >= required_memory):
            if (check_times == 3):
                print(f"GPU memory check passed: {free_gb:.2f} GB free, which meets the requirement of {required} GB. Proceeding with training...")
                break
            else:
                print(f"GPU memory check passed: {free_gb:.2f} GB free, which meets the requirement of {required} GB. ({check_times}/3)")
                check_times += 1
        else:
            if free_gb != last_free_gb:
                print(f"GPU memory insufficient: only {free_gb:.2f} GB free, which is below the required {required} GB. Retrying in 10 seconds...")
                last_free_gb = free_gb
            check_times = 0

        time.sleep(10)

    return True

def main(args):
    backbone_config = {
        "pre-prompt": "{bos_token}system\nYou are a helpful speech recognition model. Please refer to the keyword to transcribe the input speech.{eos_token}\n{bos_token}user\n[keyword]{bwords}\n[speech]",
        "post-prompt": "{eos_token}\n{bos_token}assistant\n",
    }

    model_config = {
        'adapter_model': args.adapter_model,
        'connector_model': args.connector_model,
        'backbone_model': args.backbone_model,
        'feature_dim': args.feature_dim,
        'connector_dim': args.connector_dim,
        'ctc_gate_dim': args.ctc_gate_dim,
        'connector_k': args.connector_k,
        'ctc_weight': args.ctc_weight,
        'use_lora': args.use_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'max_lr': args.max_lr,
        'batch_size': args.batch_size,
        'total_training_step': args.total_training_step,
        'train_batch_per_epoch': args.train_batch_per_epoch,
        'grad_accumulate_steps': args.grad_accumulate_steps,
        'bias_mode': args.bias_mode,
        'asr_pre_prompt': backbone_config["pre-prompt"],
        'asr_post_prompt': backbone_config["post-prompt"],
        'checkpoint': args.checkpoint
    }

    print(model_config)

    model = SpeechLLM(**model_config)

    collator = ASR_Collator(model.tokenizer)

    train_dataset = Prompted_ASR_Dataset(
        data_info=args.train_dataset,
        tokenizer=model.tokenizer,
        pre_prompt=model.asr_pre_prompt,
        post_prompt=model.asr_post_prompt,
        bias_mode=model.bias_mode
    )
    valid_dataset = Prompted_ASR_Dataset(
        data_info=args.valid_dataset,
        tokenizer=model.tokenizer,
        pre_prompt=model.asr_pre_prompt,
        post_prompt=model.asr_post_prompt,
        bias_mode=model.bias_mode
    )

    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=model.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=model.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )

    if model.checkpoint:
        model.load_state_dict(
            torch.load(model.checkpoint, map_location='cpu')['state_dict'], strict=True
        )
        print(f"Loaded checkpoint from {model.checkpoint}")

    save_path = get_exp_path()
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}',
        save_top_k=3,
        monitor="valid/wer",
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor="valid/loss",
        min_delta=0.00,
        patience=args.early_stop_patience,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.wandb:
        logger = WandbLogger(
            project=args.wandb,
            name=save_path,
            save_dir=save_path,
            config=model_config
        )
    else:
        logger = TensorBoardLogger(
            save_dir=save_path,
            name="tensorboard_logs"
        )

    trainer = Trainer(
        max_epochs=model.max_epochs,
        accelerator="gpu", devices=args.num_device,
        limit_train_batches=model.train_batch_per_epoch,
        limit_val_batches=len(valid_dataset),
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        fast_dev_run=False,
        accumulate_grad_batches=model.grad_accumulate_steps,
        logger=logger
    )

    check_VRAM(22)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=model.checkpoint)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dataset', default="./dataset/LibriSpeech/train-960h.json", help='the json file (including feature path) for training.')
    parser.add_argument('--valid-dataset', default="./dataset/LibriSpeech/dev-clean.json", help='the json file (including feature path) for validation.')
    parser.add_argument('--checkpoint', default=None, type=str, help='the path of checkpoint (i.e., ./exp/01/best.ckpt)')
    parser.add_argument('--bias-mode', default='train', type=str, help='training or inference.')

    parser.add_argument('--adapter-model', default='conv-branchformer', type=str)
    parser.add_argument('--connector-model', default='cnn', type=str)
    parser.add_argument('--backbone-model', default='HuggingFaceTB/SmolLM2-135M-Instruct', type=str, help='backbone model repository name or local path (e.g., HuggingFaceTB/SmolLM-135M-Instruct).')

    parser.add_argument('--feature-dim', default=1280, type=int, help='the output size of audio encoder model.')
    parser.add_argument('--connector-dim', default=512, type=int, help='the hidden size of audio adapter and connector model.')
    parser.add_argument('--ctc-gate-dim', default=128, type=int, help='the hidden size of CTC gate module.')

    parser.add_argument('--connector-k', default=4, type=int)
    parser.add_argument('--ctc-weight', default=0.5, type=float, help='the weight of CTC loss.')
    parser.add_argument('--use-lora', action='store_true', help='whether to use LoRA in the backbone model.')
    parser.add_argument('--lora-r', default=8, type=int, help='the LoRA rank.')
    parser.add_argument('--lora-alpha', default=32, type=int, help='the LoRA alpha.')
    parser.add_argument('--max-lr', default=1e-4, type=float, help='the max learning rate.')

    parser.add_argument('--total-training-step', default=100000, type=int, help='the total training steps.')
    parser.add_argument('--batch-size', default=8, type=int, help='the training batch size.')
    parser.add_argument('--train-batch-per-epoch', default=2500, type=int, help='the number of training batches per epoch.')
    parser.add_argument('--grad-accumulate-steps', default=4, type=int, help='the number of gradient accumulation steps.')
    parser.add_argument('--early-stop-patience', default=10, type=int, help='the patience epochs for early stopping.')

    parser.add_argument('--num-device', default=1, type=int, help='the number of devices')
    parser.add_argument('--wandb', default=None, type=str, help='the name of wandb project.')

    args = parser.parse_args()

    main(args)
