
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
    data_path = "data/training.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded dataset with {len(text):,} characters.")
    
    # Create and train tokenizer
    tokenizer = rbt.BpeTokenizer(num_merges=100)
    print("Training tokenizer...")
    tokenizer.fit(text)
    
    # Ensure checkpoints folder exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save vocab via Rust helper
    vocab_path = "checkpoints/tokenizer_vocab.json"
    rbt.save_vocab(tokenizer, vocab_path)
    
    # Save config
    vocab_size = tokenizer.get_vocab_size()
    config = {"vocab_size": vocab_size, "num_merges": tokenizer.get_num_merges()}
    config_path = "checkpoints/tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save tokenized dataset
    tokenized_path = "checkpoints/tokenized_training.json"
    token_ids = tokenizer.encode(text)
    rbt.save_tokenized_text(token_ids, tokenized_path)
    
    elapsed = time.time() - start_time
    print(f"\nTrained BPE tokenizer with vocab size {vocab_size}")
    print(f"Saved vocab to {vocab_path}")
    print(f"Saved config to {config_path}")
    print(f"Saved tokenized training text to {tokenized_path}")
    print(f"Training took {elapsed:.2f} seconds")


if __name__ == "__main__":
    train_tokenizer()