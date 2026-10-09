# Analysis of Twitter Bot Usage for Indian Political Issues

A research project template for studying bot-like activity in Twitter/X discussions related to Indian political issues, including discussion around the Citizenship (Amendment) Act (CAA).

> **Important:** This repository is a starter structure. Add your actual dataset, research notes, trained model, evaluation metrics, and verified findings before presenting it as completed work. Do not commit API keys, private data, or unlicensed datasets.

## Objectives
- Prepare and clean social-media text data for analysis.
- Extract text-based features for classification.
- Train a Relevance Vector Machine (RVM) classifier when labelled data is available.
- Evaluate performance using suitable classification metrics.
- Document observed patterns without overstating what the data can prove.

## Repository structure

```text
twitter-bot-analysis/
├── data/
│   └── README.md
├── notebooks/
│   └── (add exploratory analysis notebooks here)
├── reports/
│   └── (add figures and your final report here)
├── src/
│   ├── preprocess.py
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset format

Prepare a CSV file such as `data/tweets.csv` with:
- `text`: tweet/post text
- `label`: ground-truth label, e.g. `bot` or `human`

Use labels only if they come from a documented, defensible labelling process. If you only have unlabelled data, perform exploratory analysis instead of supervised training. Do not infer that an account is a bot from political viewpoint.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

Place your labelled dataset at `data/tweets.csv`, then run:

```bash
python src/train_model.py --input data/tweets.csv --text-column text --label-column label
```

The script uses TF-IDF features and scikit-learn's RBF Support Vector Classifier as a runnable baseline. **This is not an RVM implementation.** Replace it with a validated RVM package/implementation if your original research specifically used a Relevance Vector Machine, and document the exact method.

## Suggested report sections
1. Research question and scope
2. Data source, collection period, and API/terms compliance
3. Data cleaning and feature engineering
4. Labelling method and limitations
5. Model design and train/test split
6. Metrics: precision, recall, F1-score, confusion matrix
7. Error analysis and limitations
8. Findings, references, and reproducibility notes

## Responsible research
- Follow the platform's current terms and applicable laws.
- Remove personal or sensitive information where possible.
- Keep credentials in environment variables; never commit secrets.
- Report uncertainty and false positives. Model predictions are not proof that an account is automated.
- Avoid making claims about political groups or individuals that the data cannot substantiate.
