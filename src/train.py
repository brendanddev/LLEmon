
""" 
train.py

Defines the training loop for the Transformer model on character-level text data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from utils.data_utils import load_data, get_batch, load_rust_tokenizer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1000
    lr = 3e-4
    eval_interval = 100

    # Load data
    # This now loads your Rust-trained tokenizer
    tokenizer, train_data, val_data, block_size, batch_size = load_data(
        block_size=256, verbose=True
    )

    # Initialize model with Rust tokenizer vocab size
    vocab_size = len(tokenizer.token2id)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=128,
        N=2,
        heads=8,
        d_ff=512,
        max_len=block_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for step in range(1, epochs + 1):
        model.train()
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch(val_data, block_size, batch_size, device)
                val_logits = model(xb)
                val_loss = criterion(val_logits.view(-1, val_logits.size(-1)), yb.view(-1))
            print(f"Step {step} | Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")

    # Save model and tokenizer
    torch.save({
        "model_state": model.state_dict(),
        "vocab": list(tokenizer.token2id),  # save Rust tokenizer vocab
    }, "checkpoints/model.pth")
    print("Model and tokenizer saved to model.pth")


if __name__ == "__main__":
    train()
