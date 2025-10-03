
""" 
data_utils.py

Defines utility functions for loading and batching text data for training.
"""

import torch
from tokenizer.bpetokenizer import BPETokenizer
    
def load_data(path="data/training.txt", train_split=0.9, block_size=256, batch_size=32, verbose=False):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        
    # Build tokenizer and encode text
    tokenizer = BPETokenizer(num_merges=1000)
    tokenizer.fit(text)

    # Encode entire text dataset into token IDs
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    split_idx = int(train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    if verbose:
        print_stats(tokenizer, train_data, val_data, block_size, batch_size)
    
    return tokenizer, train_data, val_data, block_size, batch_size
    

def get_batch(data, block_size, batch_size, device="cpu"):
    # Sample random indices
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def print_stats(tokenizer, train_data, val_data, block_size, batch_size):
    print(f"Total characters in dataset: {len(train_data) + len(val_data)}")
    print(f"Unique characters (vocab size): {tokenizer.vocab_size}")
    print(f"Training split: {len(train_data)} characters")
    print(f"Validation split: {len(val_data)} characters")
    print(f"Block size: {block_size}")
    print(f"Batch size: {batch_size}")