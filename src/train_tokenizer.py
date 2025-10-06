
""" 
train_tokenizer.py

Trains a BPE tokenizer on the provided text dataset and saves the vocabulary size.
"""

import rust_bpe_tokenizer as rbt
import json
import os
import time

def train_tokenizer():
    start_time = time.time()

    # Load training data
    with open("data/training.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded dataset with {len(text):,} characters.")
    
    # Create and train tokenizer
    tokenizer = rbt.BpeTokenizer(num_merges=1000)
    print("Training tokenizer...")
    tokenizer.fit(text)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save vocab via Rust helper
    rbt.save_vocab(tokenizer, "checkpoints/tokenizer_vocab.json")
    
    vocab_size = len(tokenizer.token2id)
    config = {"vocab_size": vocab_size, "num_merges": tokenizer.num_merges}
    with open("checkpoints/tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTrained BPE tokenizer with vocab size {vocab_size}")
    print(f"Saved vocab to checkpoints/tokenizer_vocab.json")
    print(f"Training took {elapsed:.2f} seconds")

if __name__ == "__main__":
    train_tokenizer()