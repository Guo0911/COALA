import os
import json
import time
import torch
import argparse

import pandas as pd
import soundfile as sf

from tqdm import tqdm
from transformers import WhisperProcessor
from transformers.models.whisper.modeling_whisper import WhisperModel

from models import biasLM as SpeechLLM

def get_exp_path(save_path='./exp/Inference'):
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

class COALA(SpeechLLM):
    def __init__(
        self,
        hyper_parameters,
        run_asr: bool = True,
        run_bias: bool = True,
        load_encoder: bool = False
    ):
        super().__init__(**hyper_parameters)

        self.run_asr = run_asr
        self.run_bias = run_bias

        self.load_encoder = load_encoder
        if self.load_encoder:
            self.encoder = WhisperModel.from_pretrained("openai/whisper-large-v2").encoder.eval()
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        device: str="cuda",
        strict: bool = True,
        run_asr: bool = True,
        run_bias: bool = True,
        load_encoder: bool = False
    ) -> "COALA":
        ckpt_model = torch.load(checkpoint, map_location="cpu", weights_only=False,)

        ckpt_model["hyper_parameters"]["scoring_loss"] = "DPD-loss"

        model = cls(ckpt_model["hyper_parameters"], run_asr, run_bias, load_encoder)

        model.setup_model_parameters()
        model.load_state_dict(ckpt_model["state_dict"], strict=strict)

        model.to(device)
        model.eval()

        return model

    def prompt_tokenize(
        self,
        audio_feats: torch.Tensor,
        pre_prompt: str,
        post_prompt: str,
        device: str
    ):
        audio_feats = audio_feats.to(device)

        pre_tokenized_ids = self.tokenizer(pre_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        post_tokenized_ids = self.tokenizer(post_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        return audio_feats, pre_tokenized_ids, post_tokenized_ids

    def extract_feature(
        self,
        audio_path: str,
        feats_path: str = None
    ):
        audio, sr = sf.read(audio_path)

        if self.load_encoder:
            processed_audio = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.encoder.device)
            extracted_feature = self.encoder(processed_audio).last_hidden_state
        else:
            extracted_feature = torch.from_numpy(torch.load(feats_path, weights_only=False)).float()

        audio_duration = len(audio) / sr

        return extracted_feature, audio_duration

    def scoring_bias_list(
        self,
        audio_feats,
        pre_tokenized_ids,
        post_tokenized_ids,
        bias_list,
        batch_size=200
    ):
        bias_list = ["<unbiased>"] + bias_list
        bias_scores = {}

        # process audio features and text prompts to get prefix key-value and masks
        audio_prefix_kv, audio_prefix_mask = self.prefix_process(audio_feats, pre_tokenized_ids, post_tokenized_ids)

        # batch processing for scoring bias candidates
        num_candidates = len(bias_list)
        for batch_start_idx in range(0, num_candidates, batch_size):
            batch_end_idx = min((batch_start_idx + batch_size), num_candidates)

            batch_candidates = bias_list[batch_start_idx: batch_end_idx]

            token_log_probs, entity_masks = self.calculate_entity_scores(audio_prefix_kv, audio_prefix_mask, batch_candidates)

            L_i = entity_masks.sum(dim=1)
            s_i = token_log_probs.sum(dim=1) / L_i

            for i, candidate in enumerate(batch_candidates):
                bias_scores[candidate] = s_i[i].item()

        return bias_scores

    def Transcribe(
        self,
        audio_feats,
        pre_tokenized_ids,
        post_tokenized_ids,
        transcription_config
    ):
        # audio feature to embedding (called as audio tokens)
        audio_embeds = self.connector(audio_feats)
        audio_embeds = self.adapter(audio_embeds)
        audio_embeds_ctc, ctc_logits, ctc_gate_logits = self.ctc(audio_embeds)

        # text prompt to embedding
        embedder = self.backbone.base_model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)

        # concatenate all embeddings
        combined_embeds = torch.cat([pre_prompt_embeds, audio_embeds_ctc, post_prompt_embeds], dim=1)
        attention_masks = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        # model generation
        output = self.backbone.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_masks,
            max_new_tokens=transcription_config["max_new_tokens"],
            num_beams=transcription_config["num_beams"],
            do_sample=transcription_config["do_sample"],
            min_length=transcription_config["min_length"],
            temperature=transcription_config["temperature"],
            top_k=transcription_config["top_k"],
            top_p=transcription_config["top_p"],
            repetition_penalty=transcription_config["repetition_penalty"],
            no_repeat_ngram_size=transcription_config["no_repeat_ngram_size"],
            length_penalty=transcription_config["length_penalty"],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0] # only one sample in inference

        return output_text

def main(args):
    transcription_config = {
        'max_new_tokens': args.max_new_tokens,
        'num_beams': args.num_beams,
        'do_sample': args.do_sample,
        'min_length': args.min_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'length_penalty': args.length_penalty,
    }

    data = pd.read_json(args.eval_dataset)

    bias_lists = {}
    with open(f'{args.bias_lists}/{os.path.basename(args.eval_dataset)}l', 'r') as f:
        for line in f:
            if not line.strip():
                continue

            bias_lists.update(json.loads(line))

    model = COALA.from_checkpoint(
        args.checkpoint, device="cuda", strict=False,
        run_asr=args.run_asr, run_bias=args.run_bias,
        load_encoder=args.load_encoder
    )

    save_path = get_exp_path()
    os.makedirs(f"{save_path}/results", exist_ok=True)
    with open(f"{save_path}/config.json", "w", encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    time_record = {"asr": [], "bias": [], "total": [], "duration": []}
    for data_id, data_info in tqdm(data.items(), total=len(data.keys()), desc="Inference"):
        bias_list = bias_lists[data_id]

        with torch.no_grad():
            total_start_time = time.perf_counter()

            audio_feats, audio_duration = model.extract_feature(data_info["audio"], data_info["feats"])

            if model.run_bias:
                model.backbone.set_adapter("scoring")
                model.backbone.base_model.set_adapter("scoring")

                bias_start_time = time.perf_counter()

                audio_feats, pre_tokenized_ids, post_tokenized_ids = model.prompt_tokenize(
                    audio_feats,
                    model.bias_pre_prompt.format(bos_token=model.tokenizer.bos_token, eos_token=model.tokenizer.eos_token),
                    model.bias_post_prompt.format(bos_token=model.tokenizer.bos_token, eos_token=model.tokenizer.eos_token),
                    model.device
                )
                bias_scores = model.scoring_bias_list(
                    audio_feats,
                    pre_tokenized_ids,
                    post_tokenized_ids,
                    bias_list,
                    batch_size=args.batch_size
                )
                selected_entities = sorted(bias_scores, key=bias_scores.get, reverse=True)[:args.num_selected_entity]

                bias_end_time = time.perf_counter()
            else:
                selected_entities = data_info["blist"] if data_info["blist"] else ['<unbiased>']
                bias_start_time = 0.0
                bias_end_time = -0.1

            if model.run_asr:
                model.backbone.set_adapter("default")
                model.backbone.base_model.set_adapter("default")

                asr_start_time = time.perf_counter()

                bias_list_format = f"{{{', '.join(selected_entities)}}}"
                audio_feats, pre_tokenized_ids, post_tokenized_ids = model.prompt_tokenize(
                    audio_feats,
                    model.asr_pre_prompt.format(bos_token=model.tokenizer.bos_token, eos_token=model.tokenizer.eos_token, bwords=bias_list_format),
                    model.asr_post_prompt.format(bos_token=model.tokenizer.bos_token, eos_token=model.tokenizer.eos_token, bwords=bias_list_format),
                    model.device
                )
                output_text = model.Transcribe(
                    audio_feats,
                    pre_tokenized_ids,
                    post_tokenized_ids,
                    transcription_config
                )

                asr_end_time = time.perf_counter()
            else:
                asr_start_time = 0.0
                asr_end_time = 0.0

            total_end_time = time.perf_counter()

        results = {
            "reference": data_info["words"],
            "hypothesis": output_text if model.run_asr else None,
            "bias_list": bias_list if model.run_bias else None,
            "bias_scores": bias_scores if model.run_bias else None
        }

        with open(f"{save_path}/results/{data_id}.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        time_record["asr"].append(asr_end_time - asr_start_time)
        time_record["bias"].append(bias_end_time - bias_start_time)
        time_record["total"].append(total_end_time - total_start_time)
        time_record["duration"].append(audio_duration)

    rtf_results = {
        "ASR": round(sum(time_record['asr']) / sum(time_record['duration']), 4),
        "Bias": round(sum(time_record['bias']) / sum(time_record['duration']), 4),
        "Pipeline": round(sum(time_record['total']) / sum(time_record['duration']), 4)
    }

    with open(f"{save_path}/RTF.json", "w", encoding='utf-8') as f:
        json.dump(rtf_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-asr', action="store_true")
    parser.add_argument('--run-bias', action="store_true")
    parser.add_argument('--load-encoder', action="store_true")

    parser.add_argument('--eval-dataset', default="./dataset/LibriSpeech/dev-clean.json", type=str)
    parser.add_argument('--bias-lists', default="./dataset/LibriSpeech/bias-lists/N=5000", type=str)

    parser.add_argument('--checkpoint', default=None, type=str)

    # hyper-parameters for contextual biasing
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--num-selected-entity', default=10, type=int)

    # hyper-parameters for transcription
    parser.add_argument('--max-new-tokens', default=300, type=int, help='maximum number of new tokens to generate')
    parser.add_argument('--num-beams', default=20, type=int, help='number of beams for beam search')
    parser.add_argument('--do-sample', default=True, type=bool, help='whether to use sampling for generation')
    parser.add_argument('--min-length', default=1, type=int, help='minimum length of the generated sequence')
    parser.add_argument('--temperature', default=0.8, type=float, help='temperature for sampling')
    parser.add_argument('--top-k', default=20, type=int, help='top-k sampling parameter')
    parser.add_argument('--top-p', default=0.5, type=float, help='top-p (nucleus) sampling parameter')
    parser.add_argument('--repetition-penalty', default=1.1, type=float, help='repetition penalty for generation')
    parser.add_argument('--no-repeat-ngram-size', default=3, type=int, help='no repeat ngram size for generation')
    parser.add_argument('--length-penalty', default=0.5, type=float, help='length penalty for beam search')

    args = parser.parse_args()

    main(args)
