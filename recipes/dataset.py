import json
import torch
import random

import pandas as pd

from typing import Literal
from torch.utils.data import Dataset

class ASR_Dataset(Dataset):
    def __init__(
        self,
        data_info,
        bias_mode: Literal["train", "infer"] = None,
        bias_predi: str = None
    ):
        self.data_frame = pd.read_json(data_info) # The json file created from extract script (scripts/extract_feature2pt.py).
        self.data_idx = list(self.data_frame.keys())

        self.bias_mode = bias_mode
        if self.bias_mode == "infer":
            if bias_predi is None: # bias_predi is a file used to record selected entities for each utterance, which is built by the prediction code.
                raise ValueError("bias_predi file missed")

            with open(bias_predi, 'r') as f:
                self.selected_entities = json.load(f)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        name = self.data_idx[idx] # get utterance name
        data = self.data_frame[name]

        audio_feat = torch.from_numpy(
            torch.load(data['feats'], weights_only=False)).float().squeeze()

        reference = data["words"].lower()

        if self.bias_mode == "train":
            positive = data["blist"]
            if len(positive) > 5:
                positive = random.sample(positive, 5)

            negative_pool = set()
            for random_idx in random.sample(range(len(self.data_idx)), 10):
                negative_pool.update(self.data_frame[self.data_idx[random_idx]]["blist"])

            negative = list(negative_pool)
            if len(negative) > (9 - len(positive)):
                negative = random.sample(negative, (9 - len(positive)))

            bias_list = [w.lower() for w in positive] + ['<unbiased>'] + [w.lower() for w in negative]

        elif self.bias_mode == "infer":
            bias_list = [w.lower() for w in self.selected_entities[name]] + ['<unbiased>']

        random.shuffle(bias_list)

        return name, audio_feat, reference, bias_list

class Prompted_ASR_Dataset(ASR_Dataset):
    def __init__(
        self,
        data_info,
        tokenizer,
        pre_prompt: str,
        post_prompt: str,
        bias_mode: str = None,
        bias_predi: str = None
    ):
        super().__init__(data_info, bias_mode, bias_predi)

        self.tokenizer = tokenizer
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt

    def __getitem__(self, idx):
        name, audio_feat, reference, bias_list = super().__getitem__(idx)

        bias_list = f"{{{', '.join(bias_list)}}}"

        pre_prompt = self.pre_prompt.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token, bwords=bias_list)
        post_prompt = self.post_prompt.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token, bwords=bias_list)

        reference_prompt = f"{reference}{self.tokenizer.eos_token}"

        return name, audio_feat, bias_list, pre_prompt, post_prompt, reference_prompt

class ASR_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        names = [b[0] for b in batch]

        audio_feats = torch.stack([b[1] for b in batch])

        bias_lists = [b[2] for b in batch]

        pre_prompt = [b[3] for b in batch]
        pre_tokenized_ids = self.tokenizer(pre_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

        post_prompt = [b[4] for b in batch]
        post_tokenized_ids = self.tokenizer(post_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

        reference_prompt = [b[5] for b in batch]
        reference_tokenized_ids = self.tokenizer(reference_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

        return names, audio_feats, bias_lists, pre_tokenized_ids, post_tokenized_ids, reference_tokenized_ids

class Bias_Dataset(Dataset):
    def __init__(
        self,
        data_info,
        bias_list_info
    ):
        self.data_frame = pd.read_json(data_info)
        self.data_idx = list(self.data_frame.keys())

        self.bias_lists = {}
        with open(bias_list_info, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                self.bias_lists.update(json.loads(line))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        name = self.data_idx[idx]
        data = self.data_frame[name]

        audio_feat = torch.from_numpy(
            torch.load(data['feats'], weights_only=False)).float().squeeze()

        positive = data["blist"]

        negative_pool = set(self.bias_lists[name]) - set(positive)
        negative = random.sample(list(negative_pool), (120 - len(positive)))

        if len(positive) == 0:
            positive = ['<unbiased>']
        else:
            negative.append('<unbiased>')

        return name, audio_feat, positive, negative

class Prompted_Bias_Dataset(Bias_Dataset):
    def __init__(
        self,
        data_info,
        tokenizer,
        pre_prompt: str,
        post_prompt: str,
        bias_list_info: str
    ):
        super().__init__(data_info, bias_list_info)

        self.tokenizer = tokenizer
        self.pre_prompt = pre_prompt.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token)
        self.post_prompt = post_prompt.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token)

    def __getitem__(self, idx):
        name, audio_feat, positive, negative = super().__getitem__(idx)

        return name, audio_feat, positive, negative, self.pre_prompt, self.post_prompt

class Bias_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        names = [b[0] for b in batch]

        audio_feats = torch.stack([b[1] for b in batch])

        positives = [b[2] for b in batch]
        negatives = [b[3] for b in batch]

        pre_prompt = [b[4] for b in batch]
        post_prompt = [b[5] for b in batch]

        pre_tokenized_ids = self.tokenizer(pre_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        post_tokenized_ids = self.tokenizer(post_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

        return names, audio_feats, positives, negatives, pre_tokenized_ids, post_tokenized_ids
