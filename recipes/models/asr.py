import torch
import random

import pytorch_lightning as pl

from jiwer import wer
from torch.optim import AdamW
from transformers import StoppingCriteria

from .modules import get_connector, get_adapter, get_backbone, CTC

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      StoppingCriteria.__init__(self),
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

class SpeechLLM(pl.LightningModule):
    def __init__(
        self,
        adapter_model="conv-branchformer",
        connector_model='cnn',
        backbone_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        feature_dim=1280,
        connector_dim=576,
        ctc_gate_dim=128,
        connector_k=4,
        ctc_weight=0.5,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        max_lr=1e-4,
        batch_size=1,
        total_training_step=210000,
        train_batch_per_epoch=7000,
        grad_accumulate_steps=4,
        bias_mode=None,
        asr_pre_prompt=None,
        asr_post_prompt=None,
        checkpoint=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer, self.backbone = get_backbone(backbone_model, use_lora, lora_r, lora_alpha)
        self.connector = get_connector(
            name=connector_model,
            input_dim=feature_dim,
            output_dim=connector_dim,
            k=connector_k
        )
        self.adapter = get_adapter(
            name=adapter_model,
            input_dim=connector_dim,
            output_dim=self.backbone.config.hidden_size,
            num_heads=4,
            num_depths=2
        )
        self.ctc = CTC(
            input_dim=self.backbone.config.hidden_size,
            hidden_dim=ctc_gate_dim,
            output_dim=self.backbone.config.hidden_size,
            vocab_size=self.tokenizer.vocab_size
        )
        self.ctc_loss = torch.nn.CTCLoss(blank=self.tokenizer.pad_token_id) # assign blank as '<|im_end|>'

        self.max_lr = max_lr
        self.batch_size = batch_size
        self.total_training_step = total_training_step
        self.train_batch_per_epoch = train_batch_per_epoch
        self.grad_accumulate_steps = grad_accumulate_steps
        self.max_epochs = (total_training_step // train_batch_per_epoch)
        self.num_validation_samples = 1000
        self.ctc_weight = ctc_weight

        self.bias_mode = bias_mode
        self.asr_pre_prompt = asr_pre_prompt
        self.asr_post_prompt = asr_post_prompt
        self.checkpoint = checkpoint

    def configure_optimizers(self):
        opt = [
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.adapter.parameters(), "lr": self.max_lr},
            {"params": self.ctc.parameters(), "lr": self.max_lr},
            {"params": self.backbone.parameters(), "lr": self.max_lr},
        ]

        optimizer = AdamW(opt, lr=self.max_lr, weight_decay=0.05, betas=(0.9,0.999))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=0, last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def encode(self, audio_feats, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids):
        batch_size = audio_feats.shape[0]

        # audio feature to embedding (called as audio tokens)
        audio_embeds = self.connector(audio_feats)
        audio_embeds = self.adapter(audio_embeds)
        audio_embeds_ctc, ctc_logits, ctc_gate_logits = self.ctc(audio_embeds)

        # text prompt to embedding
        embedder = self.backbone.base_model.model.model.embed_tokens 
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        # concatenate all embeddings and calculate attention mask
        combined_embeds = torch.cat([pre_prompt_embeds, audio_embeds_ctc, post_prompt_embeds, output_prompt_embeds], dim=1)
        attention_masks = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        # calculate the length of input tokens
        input_token_length = pre_tokenized_ids.shape[1] + audio_embeds_ctc.shape[1] + post_tokenized_ids.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)

        return combined_embeds, attention_masks, label_ids, input_token_length, audio_embeds, ctc_logits, ctc_gate_logits

    def forward(self, embeds, atts, label_ids):
        return self.backbone(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )

    def training_step(self, batch, batch_idx):
        names, audio_feats, bias_lists, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids = batch

        batch_size = audio_feats.shape[0]

        # model inference
        combined_embeds, attention_masks, label_ids, input_token_length, audio_embeds, ctc_logits, ctc_gate_logits = self.encode(
            audio_feats, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids
        )
        outputs = self.forward(combined_embeds, attention_masks, label_ids)

        # decode reference and hypothesis
        logits = outputs.logits
        predicted_ids = torch.argmax(logits[:,input_token_length:,:], dim=-1).cpu()

        hypothesis = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        references = self.tokenizer.decode(reference_tokenized_ids[0], skip_special_tokens=False)

        if batch_idx < 5:
            print('-'*20 + ' Training Sample Generation ' + '-'*20)
            print(f'utt_id: {names[0]}')
            print(f'Biasing List: {bias_lists[0]}')
            print(f'prompt:\n{self.tokenizer.decode(pre_tokenized_ids[0], skip_special_tokens=False)}{self.tokenizer.decode(post_tokenized_ids[0], skip_special_tokens=False)}')
            print(f'Reference:\n{references}')
            print(f'Hypothesis:\n{hypothesis}')
            print('-'*70)

        # compute word error rate
        wer_metric = wer(references.lower(), hypothesis.lower())
        self.log("train/wer", wer_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        # compute generation loss and ctc loss
        input_lengths = torch.tensor([feat.size(0) for feat in audio_embeds])
        target_lengths = (reference_tokenized_ids != self.tokenizer.pad_token_id).sum(dim=-1)

        ctc_log_probs = ctc_logits.log_softmax(2).transpose(0, 1) # (time, batch, vocab)
        ctc_loss = self.ctc_loss(ctc_log_probs, reference_tokenized_ids, input_lengths, target_lengths)

        ctc_gate_log_probs = ctc_gate_logits.log_softmax(2).transpose(0, 1) # (time, batch, vocab)
        ctc_gate_loss = self.ctc_loss(ctc_gate_log_probs, reference_tokenized_ids, input_lengths, target_lengths)

        ctc_loss = ctc_loss + ctc_gate_loss
        gen_loss = outputs["loss"]

        loss = gen_loss + (self.ctc_weight * ctc_loss)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/gen_loss", gen_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/ctc_loss", ctc_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        names, audio_feats, bias_lists, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids = batch

        batch_size = audio_feats.shape[0]

        # model inference
        combined_embeds, attention_masks, label_ids, input_token_length, audio_embeds, ctc_logits, ctc_gate_logits = self.encode(
            audio_feats, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids
        )
        outputs = self.forward(combined_embeds, attention_masks, label_ids)

        # decode reference and hypothesis
        logits = outputs.logits
        predicted_ids = torch.argmax(logits[:,input_token_length:,:], dim=-1).cpu()

        hypothesis = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        references = self.tokenizer.decode(reference_tokenized_ids[0], skip_special_tokens=False)

        # compute word error rate
        wer_metric = wer(references.lower(), hypothesis.lower())
        self.log("valid/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        # compute generation loss and ctc loss
        input_lengths = torch.tensor([feat.size(0) for feat in audio_embeds])
        target_lengths = (reference_tokenized_ids != self.tokenizer.pad_token_id).sum(dim=-1)

        ctc_log_probs = ctc_logits.log_softmax(2).transpose(0, 1) # (time, batch, vocab)
        ctc_loss = self.ctc_loss(ctc_log_probs, reference_tokenized_ids, input_lengths, target_lengths)

        ctc_gate_log_probs = ctc_gate_logits.log_softmax(2).transpose(0, 1) # (time, batch, vocab)
        ctc_gate_loss = self.ctc_loss(ctc_gate_log_probs, reference_tokenized_ids, input_lengths, target_lengths)

        ctc_loss = ctc_loss + ctc_gate_loss
        gen_loss = outputs["loss"]

        loss = gen_loss + (self.ctc_weight * ctc_loss)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("valid/gen_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("valid/ctc_loss", ctc_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        # output some sample generations
        if batch_idx in [0, 1, 2]:
            print('-'*20 + ' Validation Sample Generation ' + '-'*20)
            print(f'hypothesis {batch_idx}:\n{hypothesis}')
            print('-'*70)
            print(f'references {batch_idx}:\n{references}')
            print('-'*70)

        return {"val_loss": loss, "val_wer": wer_metric}

    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 1)

    def generate(self, batch, gen_dict_config, device):
        names, audio_feats, bias_lists, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids = batch

        # audio feature to embedding (called as audio tokens)
        audio_embeds = self.connector(audio_feats.to(device))
        audio_embeds = self.adapter(audio_embeds)
        audio_embeds_ctc, ctc_logits, ctc_gate_logits = self.ctc(audio_embeds)

        # text prompt to embedding
        embedder = self.backbone.base_model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids.to(device))
        post_prompt_embeds = embedder(post_tokenized_ids.to(device))

        # concatenate all embeddings
        combined_embeds = torch.cat([pre_prompt_embeds, audio_embeds_ctc, post_prompt_embeds], dim=1)
        attention_masks = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        # model generation
        output = self.backbone.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_masks,
            max_new_tokens=gen_dict_config["max_new_tokens"],
            num_beams=gen_dict_config["num_beams"],
            do_sample=gen_dict_config["do_sample"],
            min_length=gen_dict_config["min_length"],
            temperature=gen_dict_config["temperature"],
            top_k=gen_dict_config["top_k"],
            top_p=gen_dict_config["top_p"],
            repetition_penalty=gen_dict_config["repetition_penalty"],
            no_repeat_ngram_size=gen_dict_config["no_repeat_ngram_size"],
            length_penalty=gen_dict_config["length_penalty"],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return output_text
