
""" 
data_utils.py

Defines utility functions for loading and batching text data for training.
"""

import torch
import json
import rust_bpe_tokenizer as rbt


def load_rust_tokenizer(vocab_path="checkpoints/tokenizer_vocab.json", num_merges=100):
    """Load trained Rust BPE tokenizer from JSON vocab."""
    tokenizer = rbt.BpeTokenizer(num_merges=num_merges)

    # Load token2id mapping
    with open(vocab_path, "r", encoding="utf-8") as f:
        token2id = json.load(f)

    tokenizer.token2id = token2id
    tokenizer.id2token = {int(v): k for k, v in token2id.items()}
    return tokenizer


def load_data(path="data/training.txt", train_split=0.9, block_size=256, batch_size=32, verbose=False):
    """Load text data and encode using Rust BPE tokenizer."""
    # Load raw text
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    print("Loaded text length:", len(text))

    # Load pre-trained tokenizer
    tokenizer = load_rust_tokenizer(vocab_path="checkpoints/tokenizer_vocab.json", num_merges=100)
    print(f"Tokenizer vocab size: {len(tokenizer.token2id)}")

    # Encode entire dataset into token IDs
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split train/validation
    split_idx = int(train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    if verbose:
        print_stats(tokenizer, train_data, val_data, block_size, batch_size)

    return tokenizer, train_data, val_data, block_size, batch_size


def get_batch(data, block_size, batch_size, device="cpu"):
    """Sample random batches from dataset."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def print_stats(tokenizer, train_data, val_data, block_size, batch_size):
    vocab_size = len(tokenizer.token2id)
    print(f"Total tokens in dataset: {len(train_data) + len(val_data)}")
    print(f"Unique tokens (vocab size): {vocab_size}")
    print(f"Training split: {len(train_data)} tokens")
    print(f"Validation split: {len(val_data)} tokens")
    print(f"Block size: {block_size}")
    print(f"Batch size: {batch_size}")
