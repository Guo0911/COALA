import torch
import random

import pytorch_lightning as pl

from peft import LoraConfig
from torch.optim import AdamW
from transformers import StoppingCriteria
from transformers.cache_utils import DynamicCache

from .modules import get_connector, get_adapter, get_backbone, CTC, get_loss

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
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        max_lr=1e-4,
        batch_size=1,
        total_training_step=210000,
        train_batch_per_epoch=7000,
        grad_accumulate_steps=4,
        scoring_loss=None,
        asr_pre_prompt=None,
        asr_post_prompt=None,
        bias_pre_prompt=None,
        bias_post_prompt=None,
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

        self.hidden_to_score = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        self.scoring_loss = get_loss(scoring_loss)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.max_lr = max_lr
        self.batch_size = batch_size
        self.total_training_step = total_training_step
        self.train_batch_per_epoch = train_batch_per_epoch
        self.grad_accumulate_steps = grad_accumulate_steps
        self.max_epochs = (total_training_step // train_batch_per_epoch)
        self.num_validation_samples = 1000

        self.asr_pre_prompt = asr_pre_prompt
        self.asr_post_prompt = asr_post_prompt
        self.bias_pre_prompt = bias_pre_prompt
        self.bias_post_prompt = bias_post_prompt
        self.checkpoint = checkpoint

    def setup_model_parameters(self): # freeze all parameters, only update the new LoRA parameters in the backbone.
        for param in [
            *self.backbone.parameters(),
            *self.connector.parameters(),
            *self.adapter.parameters(),
            *self.ctc.parameters()
        ]:
            param.requires_grad = False

        for param in [
            *self.hidden_to_score.parameters()
        ]:
            param.requires_grad = True

        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.backbone.add_adapter("scoring", peft_config)

        self.backbone.set_adapter("scoring")
        self.backbone.base_model.set_adapter("scoring")

        for name, param in self.backbone.named_parameters():
            if "scoring" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def configure_optimizers(self):
        opt = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = AdamW(opt, lr=self.max_lr, weight_decay=0.05, betas=(0.9,0.999))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=0, last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def prefix_process(self, audio_feats, pre_tokenized_ids, post_tokenized_ids):
        # audio feature to embedding (called as audio tokens)
        audio_embeds = self.connector(audio_feats)
        audio_embeds = self.adapter(audio_embeds)
        audio_embeds_ctc, ctc_logits, ctc_gate_logits = self.ctc(audio_embeds)

        # text prompt to embedding
        embedder = self.backbone.base_model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)

        # concatenate prefix embeddings and compute kv cache, decrease computation for each entity
        audio_prefix_embeds = torch.cat([pre_prompt_embeds, audio_embeds_ctc, post_prompt_embeds], dim=1)
        audio_prefix_mask = torch.ones(audio_prefix_embeds.size()[:-1], dtype=torch.long).to(audio_prefix_embeds.device)

        outputs = self.backbone(
            inputs_embeds=audio_prefix_embeds,
            attention_mask=audio_prefix_mask,
            use_cache=True
        )

        audio_prefix_kv = outputs.past_key_values

        return audio_prefix_kv, audio_prefix_mask

    def calculate_entity_scores(self, audio_prefix_kv, audio_prefix_mask, entities):
        entity_inputs = self.tokenizer([f" {entity.lower()}" for entity in entities], padding=True, return_tensors='pt', add_special_tokens=False).to(audio_prefix_mask.device)

        entity_ids = entity_inputs["input_ids"]
        entity_masks = entity_inputs["attention_mask"]

        batch_size = entity_ids.shape[0]

        embedder = self.backbone.base_model.model.model.embed_tokens
        entity_embeds = embedder(entity_ids)

        expanded_kv = []
        for layer_k, layer_v in audio_prefix_kv:
            expanded_kv.append((
                layer_k.expand(batch_size, -1, -1, -1),
                layer_v.expand(batch_size, -1, -1, -1)
            ))
        expanded_kv = DynamicCache.from_legacy_cache(tuple(expanded_kv))
        expanded_mask = torch.cat([audio_prefix_mask.expand(batch_size, -1), entity_masks], dim=1)

        outputs = self.backbone(
            inputs_embeds=entity_embeds,
            attention_mask=expanded_mask,
            past_key_values=expanded_kv,
            use_cache=True,
            output_hidden_states=True
        )

        token_scores = self.hidden_to_score(outputs.hidden_states[-1]).squeeze(-1)
        token_scores = token_scores * entity_masks

        return token_scores, entity_masks

    def training_step(self, batch, batch_idx):
        names, audio_feats, positives, negatives, pre_tokenized_ids, post_tokenized_ids = batch

        batch_size = audio_feats.shape[0]

        audio_prefix_kv, audio_prefix_mask = self.prefix_process(audio_feats, pre_tokenized_ids, post_tokenized_ids)

        entities = positives[0] + negatives[0]
        token_log_probs, entity_masks = self.calculate_entity_scores(audio_prefix_kv, audio_prefix_mask, entities)

        L_i = entity_masks.sum(dim=1)
        s_i = token_log_probs.sum(dim=1) / L_i

        s_pos = s_i[:len(positives[0])]
        s_neg = s_i[len(positives[0]):]

        loss, pos_loss, neg_loss = self.scoring_loss(s_pos, s_neg)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/pos_loss", pos_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/neg_loss", neg_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        names, audio_feats, positives, negatives, pre_tokenized_ids, post_tokenized_ids = batch

        batch_size = audio_feats.shape[0]

        audio_prefix_kv, audio_prefix_mask = self.prefix_process(audio_feats, pre_tokenized_ids, post_tokenized_ids)

        entities = positives[0] + negatives[0]
        token_log_probs, entity_masks = self.calculate_entity_scores(audio_prefix_kv, audio_prefix_mask, entities)

        L_i = entity_masks.sum(dim=1)
        s_i = token_log_probs.sum(dim=1) / L_i

        s_pos = s_i[:len(positives[0])]
        s_neg = s_i[len(positives[0]):]

        loss, pos_loss, neg_loss = self.scoring_loss(s_pos, s_neg)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("valid/pos_loss", pos_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("valid/neg_loss", neg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        return {"val_loss": loss}

    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 1)
