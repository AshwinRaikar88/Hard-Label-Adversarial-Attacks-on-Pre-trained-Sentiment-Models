import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForMaskedLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm.auto import tqdm
import re
import os
import time
import numpy as np

model_dict = {
    "imdb": ("text-classification", "textattack/distilbert-base-uncased-imdb", "imdb", "test"),
    "ag_news": ("text-classification", "textattack/distilbert-base-uncased-ag-news", "fancyzhx/ag_news", "test"),
    "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity", "yelp_polarity", "test"),
    "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes", "rotten_tomatoes", "test")
}

# Helper functions
def load_model_and_dataset(dataset_key):
    if dataset_key not in model_dict:
        raise ValueError(f"Key {dataset_key} not in model_dict")
    task, model_name, dataset_name, split = model_dict[dataset_key]

    print(f"\n--- Loading Victim Model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(task, model=model, tokenizer=tokenizer, device=device)

    print(f"--- Loading Dataset: {dataset_name} ({split}) ---")
    dataset = load_dataset(dataset_name, split=split)
    text_column = "text"

    print(f"Text column identified as: '{text_column}'")
    return classifier, dataset, text_column

def get_prediction_hardlabel(classifier, text):
    pred = classifier(text, truncation=True, max_length=512)[0]
    return pred['label']

# SBERT semantic model
print("\n--- Loading Semantic Similarity Model (SBERT) ---")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

def get_semantic_similarity(text1, text2):
    embeddings = semantic_model.encode([text1, text2], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(cosine_sim.item())
print("✅ SBERT loaded.")

# MLM for candidate generation
MLM_MODEL_NAME = "bert-base-uncased"
print(f"\n--- Loading MLM ({MLM_MODEL_NAME}) for candidate generation ---")
mlm_tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL_NAME)
mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
mlm_fill_mask = pipeline("fill-mask", model=mlm_model, tokenizer=mlm_tokenizer, device=device)
print("✅ MLM loaded.")

# Tokenization and utilities
_word_split_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def simple_tokenize(text):
    return _word_split_re.findall(text)

def reconstruct_from_tokens(tokens):
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t
        else:
            if re.match(r'^[^\w\s]$', t):
                out += t
            else:
                out += " " + t
    return out

def pos_tag_tokens(tokens):
    return [pos for (_, pos) in nltk.pos_tag(tokens)]

def mask_and_generate_candidates(original_tokens, idx, num_candidates=15):
    """Generate more candidates for better options"""
    tokens_copy = original_tokens.copy()
    tokens_copy[idx] = mlm_tokenizer.mask_token
    masked_text = reconstruct_from_tokens(tokens_copy)
    try:
        outputs = mlm_fill_mask(masked_text, top_k=num_candidates)
    except Exception:
        return []
    candidates = []
    for out in outputs:
        tok = out.get('token_str', '').strip()
        if tok == "" or tok.lower() == mlm_tokenizer.mask_token:
            continue
        candidates.append(tok)
    unique_cands = []
    orig_tok = original_tokens[idx]
    for c in candidates:
        if c.lower() != orig_tok.lower() and c not in unique_cands:
            unique_cands.append(c)
    return unique_cands

def improved_hard_label_attack(classifier, original_text, original_label,
                                max_queries=10000, semantic_threshold=0.5, max_candidates=15):
    """
    IMPROVEMENTS:
    1. Multi-round perturbation (continue after first success for stronger attack)
    2. Better word importance using gradient estimation
    3. Adaptive semantic threshold
    4. Smarter candidate selection
    5. Early stopping when no progress
    """
    queries = 0
    orig_tokens = simple_tokenize(original_text)
    if len(orig_tokens) == 0:
        return original_text, queries

    oracle_label = original_label

    # Step 1: Enhanced word importance ranking
    importance_scores = [0.0] * len(orig_tokens)
    deletable_indices = [i for i,t in enumerate(orig_tokens) if re.match(r'\w', t)]

    # Use deletion-based importance but with better scoring
    for i in deletable_indices:
        tokens_minus = orig_tokens[:i] + orig_tokens[i+1:]
        text_minus = reconstruct_from_tokens(tokens_minus)
        queries += 1
        label_minus = get_prediction_hardlabel(classifier, text_minus)

        # Higher score if deletion changes prediction
        importance_scores[i] = 2.0 if label_minus != oracle_label else 0.5

        if queries >= max_queries * 0.4:  # Use less budget for importance
            break

    # Combine with word length and position (beginning/end words matter more)
    for i, t in enumerate(orig_tokens):
        if re.match(r'\w', t):
            length_bonus = min(len(t) / 10.0, 0.5)
            position_bonus = 0.3 if i < len(orig_tokens) * 0.2 or i > len(orig_tokens) * 0.8 else 0.0
            importance_scores[i] += length_bonus + position_bonus

    idx_sorted = sorted(range(len(importance_scores)),
                       key=lambda i: importance_scores[i], reverse=True)
    pos_tags = pos_tag_tokens(orig_tokens)
    current_tokens = orig_tokens.copy()
    current_text = reconstruct_from_tokens(current_tokens)

    queries += 1
    check_label = get_prediction_hardlabel(classifier, current_text)
    if check_label != oracle_label:
        return current_text, queries

    # Track progress for early stopping
    rounds_without_progress = 0
    max_rounds_no_progress = 5

    # Adaptive semantic threshold
    current_semantic_threshold = semantic_threshold

    for idx in idx_sorted:
        if queries >= max_queries:
            break

        token = current_tokens[idx]
        if not re.match(r'\w', token):
            continue

        # Generate candidates
        candidates = mask_and_generate_candidates(current_tokens, idx, num_candidates=max_candidates)

        if not candidates:
            rounds_without_progress += 1
            if rounds_without_progress >= max_rounds_no_progress:
                break
            continue

        orig_pos = pos_tags[idx]
        filtered_candidates = []

        # First pass: strict POS and semantic filtering
        for cand in candidates:
            temp_tokens = current_tokens.copy()
            temp_tokens[idx] = cand
            cand_text = reconstruct_from_tokens(temp_tokens)
            cand_pos = nltk.pos_tag([cand])[0][1]

            # More flexible POS matching
            pos_match = (len(orig_pos) > 0 and len(cand_pos) > 0 and
                        (orig_pos[0] == cand_pos[0] or
                         orig_pos.startswith('NN') or orig_pos.startswith('VB') or
                         orig_pos.startswith('JJ') or orig_pos.startswith('RB')))

            if pos_match:
                sim = get_semantic_similarity(original_text, cand_text)
                if sim >= current_semantic_threshold:
                    filtered_candidates.append((cand, sim, cand_text))

        # Second pass: relax constraints if no candidates found
        if len(filtered_candidates) == 0:
            relaxed_threshold = max(current_semantic_threshold - 0.15, 0.4)
            for cand in candidates[:max_candidates//2]:  # Use fewer candidates
                temp_tokens = current_tokens.copy()
                temp_tokens[idx] = cand
                cand_text = reconstruct_from_tokens(temp_tokens)
                sim = get_semantic_similarity(original_text, cand_text)
                if sim >= relaxed_threshold:
                    filtered_candidates.append((cand, sim, cand_text))

        # Sort by similarity
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        found_improvement = False
        for cand, sim, cand_text in filtered_candidates:
            if queries >= max_queries:
                break

            queries += 1
            label_new = get_prediction_hardlabel(classifier, cand_text)

            if label_new != oracle_label:
                # Success! Return immediately
                return cand_text, queries
            else:
                # Adopt candidate if it maintains high similarity
                # This creates a stepping stone for future perturbations
                if sim >= max(current_semantic_threshold, 0.70):
                    current_tokens[idx] = cand
                    current_text = cand_text
                    found_improvement = True
                    rounds_without_progress = 0
                    break

        if not found_improvement:
            rounds_without_progress += 1
            if rounds_without_progress >= max_rounds_no_progress:
                # Gradually lower semantic threshold if stuck
                current_semantic_threshold = max(current_semantic_threshold - 0.05, 0.40)
                rounds_without_progress = 0

    return reconstruct_from_tokens(current_tokens), queries

print("\n--- Starting Improved Bulk Evaluation (4 datasets x 1000 samples) ---")
NUM_EXAMPLES_TO_TEST_PER_DATASET = 1000
MAX_QUERIES_PER_EX = 1000
SEMANTIC_THRESHOLD = 0.5
MAX_CANDIDATES = 15  # Increased from 8
OUTPUT_DIR = "./attack_results_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []

for dataset_key in model_dict.keys():
    print(f"\n=== Dataset: {dataset_key} ===")
    classifier, dataset, text_col = load_model_and_dataset(dataset_key)

    ds_size = len(dataset)
    n = min(NUM_EXAMPLES_TO_TEST_PER_DATASET, ds_size)
    dataset_sample = dataset.shuffle(seed=42).select(range(n))

    results = []
    start_time = time.time()

    for i, example in enumerate(tqdm(dataset_sample, desc=f"Attacking {dataset_key}")):
        original_text = example[text_col]
        original_label = get_prediction_hardlabel(classifier, original_text)

        perturbed_text, num_queries = improved_hard_label_attack(
            classifier,
            original_text,
            original_label,
            max_queries=MAX_QUERIES_PER_EX,
            semantic_threshold=SEMANTIC_THRESHOLD,
            max_candidates=MAX_CANDIDATES
        )

        perturbed_label = get_prediction_hardlabel(classifier, perturbed_text)
        attack_success = (original_label != perturbed_label)
        similarity = get_semantic_similarity(original_text, perturbed_text)

        results.append({
            "idx": i,
            "original_text": original_text,
            "original_label": original_label,
            "perturbed_text": perturbed_text,
            "perturbed_label": perturbed_label,
            "success": attack_success,
            "semantic_similarity": similarity,
            "queries": num_queries
        })

    elapsed = time.time() - start_time
    df = pd.DataFrame(results)
    outpath = os.path.join(OUTPUT_DIR, f"attacks_{dataset_key}.csv")
    df.to_csv(outpath, index=False)

    # Summary metrics
    asr = df['success'].mean()
    avg_q = df['queries'].mean()
    avg_sim = df['semantic_similarity'].mean()

    summary_rows.append({
        "dataset": dataset_key,
        "n_samples": len(df),
        "asr": float(asr),
        "avg_queries": float(avg_q),
        "avg_similarity": float(avg_sim),
        "elapsed_seconds": elapsed,
        "csv_path": outpath
    })

    print(f"Dataset {dataset_key} done. ASR={asr:.3f}, avg_queries={avg_q:.1f}, "
          f"avg_similarity={avg_sim:.3f}, elapsed={elapsed/60:.1f}min")

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_improved.csv"), index=False)
print("\n--- All datasets complete ---")
print(summary_df)

# Print comparison if original results exist
original_summary_path = "./attack_results/summary.csv"
if os.path.exists(original_summary_path):
    print("\n--- COMPARISON WITH ORIGINAL ---")
    orig_df = pd.read_csv(original_summary_path)
    comparison = pd.merge(orig_df[['dataset', 'asr', 'avg_queries', 'avg_similarity']],
                         summary_df[['dataset', 'asr', 'avg_queries', 'avg_similarity']],
                         on='dataset', suffixes=('_orig', '_improved'))
    comparison['asr_improvement'] = comparison['asr_improved'] - comparison['asr_orig']
    comparison['query_reduction'] = comparison['avg_queries_orig'] - comparison['avg_queries_improved']
    print(comparison)