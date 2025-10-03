
""" 
data_utils.py

"""

import torch
from tokenizer.chartokenizer import CharTokenizer
    
def load_data(path="", train_split=0.9, block_size=64, batch_size=32):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        
    # Build tokenizer and encode text
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    split_idx = int(train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return tokenizer, train_data, val_data, block_size, batch_size
    

def get_batch():
    pass