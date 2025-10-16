
"""
train.py
Training loop for the Transformer model using pre-tokenized data.

Run: python -m src.utils.train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.tokenizer.chartokenizer import CharTokenizer
from src.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
from src.models.transformer import Transformer

# Generates random (input, target) batches from pre-tokenized data
def get_batch(data, block_size, batch_size, device):
    # Sample random starting positions
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)

def train():
    
    # Training parameters
    tokenizer_path = "models/hf_tokenizer.json"
    data_path = "data/processed/train_tokens.pt"
    block_size = 128
    batch_size = 32
    epochs = 2500
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    # Load tokenizer
    tokenizer = HuggingFaceTokenizer(vocab_path=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
   
    # Load pre-tokenized data
    data = torch.load(data_path)
    print(f"Loaded {len(data)} tokens from {data_path}")
   
    # Initialize the Transformer model
    model = Transformer(vocab_size=vocab_size, d_model=256, num_layers=4, heads=8, ff_hidden_dim=512)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
   
    # Training loop
    for step in range(epochs):
        xb, yb = get_batch(data, block_size, batch_size, device)
        
        # Forward pass
        logits = model(xb)
        loss = criterion(
            logits.view(-1, vocab_size),
            yb.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/transformer.pt")
    print("Training complete. Model saved to checkpoints/transformer.pt")

if __name__ == "__main__":
    train()