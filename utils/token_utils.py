
""" 
token_utils.py 
Utility functions for analyzing token statistics in a text dataset.

Brendan Dileo, August 2025
"""

from transformers import GPT2Tokenizer
from collections import Counter

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def compute_token_stats(texts):
    """ Computes token level statistics for a list of text lines. """
    all_token_lengths = []
    token_counter = Counter()

    for line in texts:
        tokens = tokenizer.encode(line)
        all_token_lengths.append(len(tokens))
        token_counter.update(tokens)

    total_tokens = sum(all_token_lengths)
    max_length = max(all_token_lengths)
    avg_length = total_tokens / len(texts)

    print(f"Number of texts: {len(texts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Max token length: {max_length}")
    print(f"Average token length: {avg_length:.2f}")
    print("Top 10 most common tokens:", token_counter.most_common(10))

    return {
        "total_tokens": total_tokens,
        "max_length": max_length,
        "avg_length": avg_length,
        "token_freq": token_counter
    }

def run_token_analysis(file_path="data/training.txt"):
    """ Runs token analysis on a text dataset file. """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    stats = compute_token_stats(texts)
    return stats