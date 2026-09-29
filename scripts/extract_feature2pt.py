import os
import json
import torch
import argparse

import soundfile as sf

from tqdm import tqdm
from transformers import WhisperProcessor
from whisper_normalizer.english import EnglishTextNormalizer
from transformers.models.whisper.modeling_whisper import WhisperModel

SUBSETS = {
    "dev-clean": ["dev-clean"],
    "dev-other": ["dev-other"],
    "test-clean": ["test-clean"],
    "test-other": ["test-other"],
    "train-960h": ["train-clean-100", "train-clean-360", "train-other-500"],
}

def feature_extraction(
    data_path: str,
    save_path: str,
    encoder: WhisperModel,
    processor: WhisperProcessor,
    normalizer: EnglishTextNormalizer,
    rare_words: set
):
    for set_name, subsets in SUBSETS.items():
        set_info = {}
        for subset in subsets:
            os.makedirs(os.path.join(save_path, 'whisper-enc-output', subset), exist_ok=True)
            speakers = os.listdir(os.path.join(data_path, subset))
            for speaker in tqdm(speakers, 'Processing {}'.format(subset)):
                chapters = os.listdir(os.path.join(data_path, subset, speaker))
                for chapter in chapters:
                    transcripts = {}
                    with open(os.path.join(data_path, subset, speaker, chapter, '{}-{}.trans.txt'.format(speaker, chapter)), 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split(' ', 1)
                            transcripts[parts[0]] = normalizer(parts[1]).upper()

                    for utt_id, utt_trans in transcripts.items():
                        current_info = {
                            "audio": os.path.join(data_path, subset, speaker, chapter, '{}.flac'.format(utt_id)),
                            "feats": os.path.join(save_path, 'whisper-enc-output', subset, '{}_enc_feats.pt'.format(utt_id)),
                            "words": utt_trans,
                            "blist": list(rare_words.intersection(set(utt_trans.split(' ')))), # find the rare words in the utterance (i.e. biasing list)
                        }

                        audio, sr = sf.read(current_info["audio"])
                        if sr != 16000:
                            raise ValueError("Sample rate mismatch: expected 16000, got {}".format(sr))

                        processed_audio = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                        extracted_feature = encoder(processed_audio.input_features.cuda()).last_hidden_state.detach().cpu().numpy()

                        torch.save(extracted_feature, current_info["feats"])

                        set_info[utt_id] = current_info

        with open(os.path.join(save_path, '{}.json'.format(set_name)), 'w', encoding='utf-8') as f:
            json.dump(set_info, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

def main(args):
    encoder = WhisperModel.from_pretrained("openai/whisper-large-v2").encoder.cuda()
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

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

    feature_extraction(
        data_path=args.data_path,
        save_path=args.save_path,
        encoder=encoder,
        processor=processor,
        normalizer=normalizer,
        rare_words=rare_words
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default='./dataset/raw/LibriSpeech', type=str, help='the path of LibriSpeech dataset.')
    parser.add_argument('--save-path', default='./dataset/LibriSpeech', type=str, help='the path to save LibriSpeech features.')

    parser.add_argument('--rare-word-path', default='./dataset/raw/bias-info/all_rare_words.txt', type=str, help='the path of LibriSpeech rare words list.')
    parser.add_argument('--common-word-path', default='./dataset/raw/bias-info/common_words_5k.txt', type=str, help='the path of LibriSpeech common words list.')

    args = parser.parse_args()

    main(args)
