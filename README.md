# Hard Label Adversarial Attacks on Pre-trained Sentiment Models
Hard-label (decision-based) adversarial attacks on pretrained sentiment and topic models using MLM-generated substitutions constrained by SBERT semantic similarity.

---

## Contents

- `attack_pipeline.py` — main script implementing the hard-label TextFooler-style attack  
- `requirements.txt` — Python dependencies   
- `attack_results/` — folder containing per-example CSVs and `summary.csv` with aggregated metrics (these results can be found in this folder **"attack results"**)

---

## Summary (results)

The experiment was run on 1,000 examples per dataset. Per-dataset CSVs were written to `./attack_results/attacks_<dataset>.csv`. The aggregated summary (and per-example CSVs) can be found in the `attack_results` folder.

| dataset | n_samples | asr | avg_queries | avg_similarity | elapsed_seconds | csv_path |
|---|---:|---:|---:|---:|---:|---|
| imdb | 1000 | 0.455 | 323.553 | 0.872839729309082 | 13518.813490390778 | `./attack_results/attacks_imdb.csv` |
| ag_news | 1000 | 0.147 | 86.841 | 0.7981941431164742 | 2534.038435935974 | `./attack_results/attacks_ag_news.csv` |
| yelp_polarity | 1000 | 0.325 | 225.928 | 0.8420404251217842 | 8907.433420658112 | `./attack_results/attacks_yelp_polarity.csv` |
| rotten_tomatoes | 1000 | 0.531 | 41.126 | 0.8109556267261505 | 830.3221998214722 | `./attack_results/attacks_rotten_tomatoes.csv` |

> All per-example CSVs and the `summary.csv` were saved into `./attack_results/` during the run — these files can be found in the folder **"attack results"**.

---

## Key features

- Black-box, **hard-label** (decision-only) attacks — only model labels are queried, no probability/gradient access required.  
- **MLM-based** candidate generation (BERT fill-mask) for context-aware substitutions.  
- **SBERT** semantic filtering to preserve meaning (cosine similarity).  
- POS-aware substitution filtering with NLTK to keep grammatical role.  
- Importance ranking via deletion tests (decision-based substitute for gradients).  
- Configurable query budget and semantic thresholds.

---

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
# .\venv\Scripts\activate     # Windows PowerShell

pip install -r requirements.txt
```

2. Install NLTK punkt/taggers (first run):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

3. Run the attack (default script name `attack_pipeline.py` — adjust if needed):

```bash
python attack_pipeline.py
```

This will:
- load victim models and datasets,
- run attacks on (by default) up to 1000 samples per dataset,
- save per-example CSVs into `./attack_results/`,
- generate a `summary.csv` summarizing ASR, average queries, average similarity, and elapsed time.

---

## Configuration

Edit constants at top of the script to change behavior:

- `NUM_EXAMPLES_TO_TEST_PER_DATASET` — number of samples per dataset (default 1000)  
- `MAX_QUERIES_PER_EX` — per-example query budget  
- `SEMANTIC_THRESHOLD` — SBERT similarity threshold for candidate acceptance  
- `MAX_CANDIDATES` — top-k MLM predictions to consider  
- `OUTPUT_DIR` — where CSVs are written (default `./attack_results`)

You can also modify `model_dict` in the script to point to different victim models and datasets.

---

## Reproducibility & Notes

- Trained models are downloaded from Hugging Face and may be cached locally (`HF_HOME` / `TRANSFORMERS_CACHE`).  
- Script uses deterministic sampling with `dataset.shuffle(seed=42)` for reproducible example selection.  
- Running this end-to-end is computationally intensive; using a GPU greatly speeds up the SBERT and MLM components.  
- Query counts include both deletion importance tests and candidate evaluation calls.
