
""" 
train_tokenizer.py

Trains a BPE tokenizer on the provided text dataset and saves the vocabulary size.
"""

import rust_bpe_tokenizer as rbt
import json
import os

def train_tokenizer():
    
    # Load training data
    with open("data/training.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded dataset with {len(text):,} characters.")
    
    tokenizer = rbt.BpeTokenizer(num_merges=1000)
    tokenizer.fit(text)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    vocab = {"vocab_size": tokenizer.get_vocab_size()}
    with open("checkpoints/tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    
    print(f"Trained BPE tokenizer with vocab size {tokenizer.get_vocab_size()} and saved to checkpoints/tokenizer_config.json")

if __name__ == "__main__":
    train_tokenizer()