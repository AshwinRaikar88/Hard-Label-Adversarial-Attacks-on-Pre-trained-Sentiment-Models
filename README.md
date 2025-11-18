# Hard Label Adversarial Attacks on Pre-trained Sentiment Models
Hard-label (decision-based) adversarial attacks on pretrained sentiment and topic models using MLM-generated substitutions constrained by SBERT semantic similarity.

---

## Contents

- `attack_pipeline.py` — main script implementing the hard-label TextFooler-style attack  
- `requirements.txt` — Python dependencies   
- `attack_results/` — folder containing per-example CSVs and `summary.csv` with aggregated metrics (these results can be found in this folder **"attack results"**)

---

## Summary (Results)

We present comprehensive results across three methodologies: **TextFooler** (gradient-based baseline), **Old Hard-Label Attack** (Project Update 1), and **Improved Hard-Label Attack** (current implementation). All experiments were conducted on 1,000 examples per dataset.

### Comparative Results Across All Methods

| Dataset | Method | ASR (%) | Avg Queries | Avg Similarity | Time (min) |
|---------|--------|---------|-------------|----------------|------------|
| **Rotten Tomatoes** | TextFooler | 50.5 | 69.6 | 0.834 | 11.3 |
| | Old Hard-Label | 53.1 | 41.1 | 0.811 | 13.8 |
| | **Improved Hard-Label** | **50.9** | **42.4** | **0.791** | **---** |
| **IMDB** | TextFooler | 41.0 | 661.7 | 0.837 | 185.5 |
| | Old Hard-Label | 45.5 | 323.6 | 0.873 | 225.3 |
| | **Improved Hard-Label** | **42.2** | **312.8** | **0.888** | **---** |
| **AG News** | TextFooler | 13.2 | 169.3 | 0.815 | 30.0 |
| | Old Hard-Label | 14.7 | 86.8 | 0.798 | 42.2 |
| | **Improved Hard-Label** | **14.7** | **83.6** | **0.796** | **---** |
| **Yelp Polarity** | TextFooler | 24.5 | 524.6 | 0.819 | 134.1 |
| | Old Hard-Label | 32.5 | 225.9 | 0.842 | 148.5 |
| | **Improved Hard-Label** | **29.3** | **223.4** | **0.855** | **---** |
| **Average** | TextFooler | 32.3 | 356.3 | 0.826 | 90.2 |
| | Old Hard-Label | 36.5 | 169.4 | 0.831 | 107.5 |
| | **Improved Hard-Label** | **34.3** | **165.6** | **0.833** | **---** |

### Key Findings

- **Query Efficiency**: Our improved hard-label method achieves **53.5% query reduction** (165.6 vs. 356.3 average queries) compared to TextFooler baseline
- **Semantic Preservation**: Progressive improvement across methods: TextFooler (0.826) → Old Hard-Label (0.831) → Improved Hard-Label (0.833)
- **Dataset-Specific Performance**:
  - **Rotten Tomatoes**: Highest ASR across all methods (50.5-53.1%), minimal queries required (41-70)
  - **IMDB**: Best semantic similarity with improved method (0.888), 52.7% query reduction vs. TextFooler
  - **AG News**: Most challenging (13.2-14.7% ASR), but still 50% query reduction vs. TextFooler
  - **Yelp Polarity**: 57.4% query reduction vs. TextFooler, excellent semantic preservation (0.855)

### Improved Hard-Label Attack Details

The current implementation incorporates several key enhancements:

| Dataset | n_samples | ASR | Avg Queries | Avg Similarity | CSV Path |
|---------|-----------|-----|-------------|----------------|----------|
| rotten_tomatoes | 1000 | 0.509 | 42.360 | 0.790831 | `./attack_results/attacks_rotten_tomatoes.csv` |
| imdb | 1000 | 0.422 | 312.807 | 0.888499 | `./attack_results/attacks_imdb.csv` |
| ag_news | 1000 | 0.147 | 83.589 | 0.796450 | `./attack_results/attacks_ag_news.csv` |
| yelp_polarity | 1000 | 0.293 | 223.414 | 0.854640 | `./attack_results/attacks_yelp_polarity.csv` |

### TextFooler Baseline Results

For comparison, we include TextFooler baseline results:

| Dataset | n_samples | ASR | Avg Queries | Avg Similarity | Time (min) |
|---------|-----------|-----|-------------|----------------|------------|
| rotten_tomatoes | 1000 | 0.505 | 69.6 | 0.834 | 11.3 |
| imdb | 1000 | 0.410 | 661.7 | 0.837 | 185.5 |
| ag_news | 1000 | 0.132 | 169.3 | 0.815 | 30.0 |
| yelp_polarity | 1000 | 0.245 | 524.6 | 0.819 | 134.1 |

### Method Evolution: Update 1 → Improved

**Improvements in Current Version:**
- ✅ MLM-guided candidate generation (replacing WordNet synonyms)
- ✅ Query-budget-aware importance scoring (60% allocation cap)
- ✅ Adaptive constraint relaxation (POS loosening, threshold reduction)
- ✅ Conservative adoption strategy for boundary-approaching
- ✅ Superior semantic preservation (0.833 vs. 0.831 average similarity)
- ✅ Better query efficiency (165.6 vs. 169.4 average queries)

**Trade-offs:**
- Slightly lower ASR (34.3% vs. 36.5%) reflecting conservative perturbation strategy
- Prioritizes semantic coherence and imperceptibility over raw attack success

> **Note**: All per-example CSVs and summary files are saved in `./attack_results/` and `./textfooler_results/` directories.

---

## Experimental Configuration

- **Query Budget**: 1,000 queries per example
- **Semantic Similarity Threshold**: τ = 0.5
- **MLM Candidates**: k = 8 top predictions
- **Importance Scoring Budget**: 60% of total queries
- **Models**: Fine-tuned DistilBERT classifiers (TextAttack hub)
- **Semantic Similarity**: Sentence-BERT (all-MiniLM-L6-v2)
- **Random Seed**: 42 (for reproducibility)

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
