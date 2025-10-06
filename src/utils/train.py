
""" 
train.py

A simple character-level training loop for the Transformer model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.tokenizer.chartokenizer import CharTokenizer
from src.models.transformer import Transformer

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Generates random (input, target) batches for next-character prediction.
def get_batch(data, block_size, batch_size, tokenizer, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(tokenizer.encode(data[i:i+block_size])) for i in ix])
    y = torch.stack([torch.tensor(tokenizer.encode(data[i+1:i+block_size+1])) for i in ix])
    return x.to(device), y.to(device)

def train():
    data_path = "data/training.txt"
    block_size = 64
    batch_size = 16
    epochs = 1000
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and tokenize data
    text = load_data(data_path)
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Initialize the Transformer model
    model = Transformer(vocab_size=vocab_size, d_model=128, num_layers=2, heads=4, ff_hidden_dim=256)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for step in range(epochs):
        xb, yb = get_batch(text, block_size, batch_size, tokenizer, device)

        # Should have shape (B, T, vocab_size)
        logits = model(xb)
        loss = criterion(
            logits.view(-1, vocab_size),
            yb.view(-1)
        )

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