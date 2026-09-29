# COALA (Contextualized ASR Leveraging Biasing Scoring)

The paper has been accepted at [INTERSPEECH 2026](https://interspeech2026.org/en-AU) and is available on [ISCA Archive](https://www.isca-archive.org/interspeech_2026/guo26b_interspeech.html) and [arXiv:2607.08117](https://arxiv.org/abs/2607.08117).

COALA is a robust framework designed to enhance speech-augmented language models (SLMs) in complex multi-entity scenarios.

![Model architecture](docs/architecture.png)

## Installation
```
# Create virtual python environment
conda create -n COALA python==3.11.10 --y
conda activate COALA

# Clone this repository
git clone https://github.com/Guo0911/COALA.git
cd COALA

# Install the required packages
pip install -r requirements.txt
```

## Prepare data

Download and extract the LibriSpeech dataset from [OpenSLR 12](https://www.openslr.org/12), then place it under `dataset/raw/LibriSpeech/`.

Download `all_rare_words.txt` and `common_words_5k.txt` from the [FBAI Deep Bias Repository](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias/words), then place them under `dataset/raw/bias-info/`.

```
# Create the biasing list for each utterance
python scripts/create_bias_list.py

# Extract the feature of encoder
python scripts/extract_feature2pt.py
```

## Usage
For model training:
```
bash stage_1.sh
bash stage_2.sh
```

For inference and evaluation:
```
bash inference.sh
```

## Citation
```
@inproceedings{guo26b_interspeech,
  title     = {{COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation}},
  author    = {Jhih-Rong Guo and Bi-Cheng Yan and Tien-Hong Lo and Berlin Chen},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {3101--3105},
  doi       = {10.21437/Interspeech.2026-1097},
  issn      = {2958-1796},
}
```

## Contact
If you have any comment or question, please contact [jhihrong@ntnu.edu.tw](mailto:jhihrong@ntnu.edu.tw)
