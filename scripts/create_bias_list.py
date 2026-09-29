import os
import json
import random
import argparse

from tqdm import tqdm
from typing import Dict
from whisper_normalizer.english import EnglishTextNormalizer

def create_bias_list(
    data_path: str,
    set_name: str,
    set_info: Dict,
    bias_list_sizes: list[int],
    rare_words: set
):
    file_handlers = {}
    for num in bias_list_sizes:
        save_path = os.path.join(data_path, "bias-lists", f"N={num}")

        os.makedirs(save_path, exist_ok=True)
        file_handlers[num] = open(os.path.join(save_path, f"{set_name}.jsonl"), "w", encoding='utf-8')

    try:
        max_size = max(bias_list_sizes)
        for utt_id, content in tqdm(set_info.items(), desc=f"Processing {set_name}"):
            positive = content['blist']

            negative_pool = random.sample(list(rare_words - set(positive)), max_size)
            for num in bias_list_sizes:
                bias_list = positive + random.sample(negative_pool, (num - len(positive)))

                line = json.dumps({utt_id: bias_list}, ensure_ascii=False)
                file_handlers[num].write(line + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        for num, f in file_handlers.items():
            f.flush()
            os.fsync(f.fileno())
            f.close()

def main(args):
    random.seed(args.random_seed)

    normalizer = EnglishTextNormalizer()

    common_words = set()
    with open(args.common_word_path, 'r') as f:
        for line in f.readlines():
            common_words.add(normalizer(line.strip()).upper())

    rare_words = set()
    with open(args.rare_word_path, 'r') as f:
        for line in f.readlines():
            normalized_word = normalizer(line.strip()).upper()
            if normalized_word not in common_words:
                rare_words.add(normalized_word)

    for set_info in [f for f in os.listdir(args.data_path) if f.endswith('.json')]:
        with open(os.path.join(args.data_path, set_info), "r") as f:
            data = json.load(f)

        create_bias_list(
            data_path = args.data_path,
            set_name = set_info.replace('.json', ''),
            set_info = data,
            bias_list_sizes = args.bias_list_sizes,
            rare_words = rare_words
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default='./dataset/LibriSpeech', type=str)

    parser.add_argument('--rare-word-path', default='./dataset/raw/bias-info/all_rare_words.txt', type=str)
    parser.add_argument('--common-word-path', default='./dataset/raw/bias-info/common_words_5k.txt', type=str)

    parser.add_argument('--bias-list-sizes', nargs='+', default=[500, 1000, 2000, 5000], type=int)

    parser.add_argument('--random-seed', default=42, type=int)

    args = parser.parse_args()

    main(args)
