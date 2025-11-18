import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm.auto import tqdm
import re
import os
import time
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

###############################################################################
# CONFIGURATION - CHANGE THESE TO RUN DIFFERENT EXPERIMENTS
###############################################################################
DATASET_TO_RUN = "rotten_tomatoes"  # Options: "imdb", "ag_news", "yelp_polarity", "rotten_tomatoes"
NUM_EXAMPLES_TO_TEST = 1000
MAX_QUERIES_PER_EXAMPLE = 1000
SEMANTIC_THRESHOLD = 0.5
OUTPUT_DIR = "./textfooler_results"
###############################################################################

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

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_synonyms_wordnet(word, pos_tag):
    """Get synonyms from WordNet based on word and POS tag"""
    synonyms = set()
    
    # Convert POS tag
    wn_pos = get_wordnet_pos(pos_tag)
    if wn_pos is None:
        return list(synonyms)
    
    # Get synsets
    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            # Only single-word synonyms, different from original
            if ' ' not in synonym and synonym.lower() != word.lower():
                synonyms.add(synonym)
    
    return list(synonyms)

def calculate_word_importance_textfooler(classifier, tokens, original_label):
    """
    Calculate word importance by measuring impact of deletion on prediction.
    Returns list of (index, importance_score) sorted by importance.
    """
    importance_list = []
    
    for i, token in enumerate(tokens):
        # Only consider words (not punctuation)
        if not re.match(r'\w', token):
            continue
        
        # Create text without this word
        tokens_without = tokens[:i] + tokens[i+1:]
        text_without = reconstruct_from_tokens(tokens_without)
        
        # Query model
        label_without = get_prediction_hardlabel(classifier, text_without)
        
        # If removing word changes prediction, it's important
        importance = 1.0 if label_without != original_label else 0.0
        importance_list.append((i, importance, token))
    
    # Sort by importance (descending)
    importance_list.sort(key=lambda x: x[1], reverse=True)
    
    return importance_list

def textfooler_attack(classifier, original_text, original_label, 
                     max_queries=1000, semantic_threshold=0.5, max_candidates=50):
    """
    TextFooler attack implementation.
    
    Algorithm:
    1. Calculate word importance ranking by deletion
    2. For each word (in importance order):
       a. Get synonyms from WordNet
       b. Filter by POS consistency
       c. Filter by semantic similarity
       d. Try each synonym and check if attack succeeds
    3. Return perturbed text if successful
    
    Returns: (perturbed_text, num_queries)
    """
    queries = 0
    orig_tokens = simple_tokenize(original_text)
    
    if len(orig_tokens) == 0:
        return original_text, queries
    
    # Step 1: Calculate word importance
    print(f"  Calculating word importance...")
    queries_before = queries
    
    importance_list = []
    for i, token in enumerate(orig_tokens):
        if not re.match(r'\w', token):
            continue
        
        tokens_without = orig_tokens[:i] + orig_tokens[i+1:]
        text_without = reconstruct_from_tokens(tokens_without)
        
        queries += 1
        label_without = get_prediction_hardlabel(classifier, text_without)
        
        importance = 1.0 if label_without != original_label else 0.0
        importance_list.append((i, importance, token))
        
        # Budget check for importance calculation
        if queries >= max_queries * 0.5:
            break
    
    importance_list.sort(key=lambda x: x[1], reverse=True)
    queries_importance = queries - queries_before
    print(f"  Used {queries_importance} queries for importance ranking")
    
    # If no important words found, use all words
    if all(imp[1] == 0.0 for imp in importance_list):
        importance_list = [(i, 0.0, tok) for i, tok in enumerate(orig_tokens) if re.match(r'\w', tok)]
    
    # Get POS tags
    pos_tags = nltk.pos_tag(orig_tokens)
    
    # Current state
    current_tokens = orig_tokens.copy()
    
    # Step 2: Try to replace words in importance order
    print(f"  Attempting word replacements...")
    for idx, importance, word in importance_list:
        if queries >= max_queries:
            break
        
        # Get POS tag
        pos_tag = pos_tags[idx][1]
        
        # Get synonyms from WordNet
        synonyms = get_synonyms_wordnet(word, pos_tag)
        
        if not synonyms:
            continue
        
        # Limit number of synonyms to try
        synonyms = synonyms[:max_candidates]
        
        # Try each synonym
        for synonym in synonyms:
            if queries >= max_queries:
                break
            
            # Create perturbed text
            temp_tokens = current_tokens.copy()
            temp_tokens[idx] = synonym
            perturbed_text = reconstruct_from_tokens(temp_tokens)
            
            # Check semantic similarity
            sim = get_semantic_similarity(original_text, perturbed_text)
            
            if sim < semantic_threshold:
                continue
            
            # Query the model
            queries += 1
            perturbed_label = get_prediction_hardlabel(classifier, perturbed_text)
            
            # Check if attack succeeds
            if perturbed_label != original_label:
                print(f"  ✓ Attack succeeded after {queries} queries!")
                return perturbed_text, queries
            
            # TextFooler adopts the synonym if it maintains high similarity
            # even if it doesn't flip the label (creates stepping stones)
            if sim >= 0.8:
                current_tokens[idx] = synonym
    
    # Attack failed
    final_text = reconstruct_from_tokens(current_tokens)
    return final_text, queries

###############################################################################
# MAIN EXECUTION
###############################################################################
print(f"\n{'='*80}")
print(f"TEXTFOOLER BASELINE ATTACK")
print(f"{'='*80}")
print(f"Dataset: {DATASET_TO_RUN}")
print(f"Number of examples: {NUM_EXAMPLES_TO_TEST}")
print(f"Max queries per example: {MAX_QUERIES_PER_EXAMPLE}")
print(f"Semantic threshold: {SEMANTIC_THRESHOLD}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*80}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and dataset
classifier, dataset, text_col = load_model_and_dataset(DATASET_TO_RUN)

# Sample dataset
ds_size = len(dataset)
n = min(NUM_EXAMPLES_TO_TEST, ds_size)
dataset_sample = dataset.shuffle(seed=42).select(range(n))

# Run attacks
results = []
start_time = time.time()

print(f"\nStarting attacks on {n} examples...\n")

for i, example in enumerate(tqdm(dataset_sample, desc=f"TextFooler on {DATASET_TO_RUN}")):
    original_text = example[text_col]
    original_label = get_prediction_hardlabel(classifier, original_text)
    
    print(f"\nExample {i+1}/{n}")
    print(f"Original: {original_text[:100]}...")
    print(f"Original label: {original_label}")
    
    perturbed_text, num_queries = textfooler_attack(
        classifier,
        original_text,
        original_label,
        max_queries=MAX_QUERIES_PER_EXAMPLE,
        semantic_threshold=SEMANTIC_THRESHOLD,
        max_candidates=50
    )
    
    perturbed_label = get_prediction_hardlabel(classifier, perturbed_text)
    attack_success = (original_label != perturbed_label)
    similarity = get_semantic_similarity(original_text, perturbed_text)
    
    print(f"Perturbed label: {perturbed_label}")
    print(f"Success: {attack_success}, Queries: {num_queries}, Similarity: {similarity:.3f}")
    
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

# Save results
df = pd.DataFrame(results)
outpath = os.path.join(OUTPUT_DIR, f"textfooler_{DATASET_TO_RUN}.csv")
df.to_csv(outpath, index=False)

# Calculate and display summary
asr = df['success'].mean()
avg_q = df['queries'].mean()
avg_sim = df['semantic_similarity'].mean()
successful_attacks = df[df['success'] == True]
avg_q_success = successful_attacks['queries'].mean() if len(successful_attacks) > 0 else 0

print(f"\n{'='*80}")
print(f"RESULTS SUMMARY - {DATASET_TO_RUN}")
print(f"{'='*80}")
print(f"Total examples: {len(df)}")
print(f"Attack Success Rate (ASR): {asr:.3f} ({asr*100:.1f}%)")
print(f"Average queries: {avg_q:.1f}")
print(f"Average queries (successful only): {avg_q_success:.1f}")
print(f"Average semantic similarity: {avg_sim:.3f}")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Time per example: {elapsed/len(df):.1f} seconds")
print(f"Results saved to: {outpath}")
print(f"{'='*80}\n")

# Save summary
summary = {
    "dataset": DATASET_TO_RUN,
    "n_samples": len(df),
    "asr": float(asr),
    "avg_queries": float(avg_q),
    "avg_queries_successful": float(avg_q_success),
    "avg_similarity": float(avg_sim),
    "elapsed_seconds": elapsed,
    "csv_path": outpath
}

summary_df = pd.DataFrame([summary])
summary_path = os.path.join(OUTPUT_DIR, f"summary_{DATASET_TO_RUN}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}\n")
