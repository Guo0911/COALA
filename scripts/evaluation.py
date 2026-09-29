import os
import math
import json
import argparse

from tqdm import tqdm
from typing import Literal
from whisper_normalizer.english import EnglishTextNormalizer

from score import score as WER_score

def find_latest_result(): # This function primarily finds the latest folder in the preset storage path.
    inference_default = './exp/Inference'
    if not os.path.exists(inference_default):
        raise ValueError("Can't find result path.")

    items = os.listdir(inference_default)
    folders = [os.path.join(inference_default, item) for item in items if os.path.isdir(os.path.join(inference_default, item))]

    return max(folders, key=os.path.getmtime)

def recall_score(sorted_rank, mode: Literal["topk-recall", "recall-topk"], value):
    total = len(sorted_rank)

    if mode == "topk-recall":
        num = sum(1 for x in sorted_rank if x <= value)

        return round((num / total), 4)

    elif mode == "recall-topk":
        if value > 100:
            raise "Maximum recall value is 100."

        num_selected = math.ceil(total * value / 100)

        return sorted_rank[num_selected]

def main(args):
    en_norm = EnglishTextNormalizer()

    result_path = find_latest_result() if args.result_path == None else args.result_path
    if not os.path.isdir(os.path.join(result_path, "results")):
        raise ValueError(f"Can't find 'results' folder in {result_path}.")

    hyps, refs, bias_lists, positive_rank = {}, {}, {}, []
    for item in tqdm(os.listdir(os.path.join(result_path, "results")), desc='evaluation...'):
        with open(os.path.join(result_path, "results", item), "r", encoding="utf-8") as f:
            result = json.load(f)

        reference = result["reference"]
        hypothesis = en_norm(result["hypothesis"]).upper()
        bias_list = set(result["bias_list"])
        bias_scores = result["bias_scores"]
        bias_oracle = list(bias_list.intersection(set(reference.split(' '))))

        item_id = os.path.splitext(item)[0]
        hyps[item_id] = hypothesis
        refs[item_id] = reference
        bias_lists[item_id] = bias_oracle

        sorted_scores = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)
        bias_rank = {key: i for i, (key, value) in enumerate(sorted_scores, start=1)}
        for bias_w in bias_oracle:
            positive_rank.append(bias_rank[bias_w])

    wer, u_wer, b_wer = WER_score(hyps, refs, bias_lists)

    sorted_rank = sorted(positive_rank)

    metrics = {
        "Bias": {
            "Recall@95": recall_score(sorted_rank, mode="recall-topk", value=95),
            "Recall#20": recall_score(sorted_rank, mode="topk-recall", value=20),
            "Recall#50": recall_score(sorted_rank, mode="topk-recall", value=50)
        },
        "ASR": {
            "Overall": wer,
            "Biased": b_wer,
            "Unbiased": u_wer
        }
    }

    with open(os.path.join(result_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--result-path', default=None, type=str)

    args = parser.parse_args()

    main(args)
