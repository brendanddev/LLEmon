
""" 
prepare_data.py

Converts raw text data into integer token tensors for training the Transformer model.
"""

import os
import torch
from src.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer

def prep_data():
    
    # Load tokenizer
    tokenizer = HuggingFaceTokenizer("models/hf_tokenizer.json")
    
    # Load training text
    with open("data/training.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Encode text to integer tokens
    token_ids = tokenizer.encode(text)
    print(f"Tokenized {len(token_ids)} tokens.")
    print("First 100 IDs:", token_ids[:100])
    
    # Convert to PyTorch tensor
    data = torch.tensor(token_ids, dtype=torch.long)
    os.makedirs("data/processed", exist_ok=True)
    
    # Save tensor to file
    torch.save(data, "data/processed/train_tokens.pt")
    print("Saved tokenized data to data/processed/train_tokens.pt")

if __name__ == "__main__":
    prep_data()