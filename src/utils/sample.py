
""" 
sample.py

Generates text from a trained Transformer model.
"""

import torch
from src.models.transformer import Transformer
from src.tokenizer.chartokenizer import CharTokenizer


def sample(model, tokenizer, start_text="h", length=100, device="cpu"):
    model.eval()
    context = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)

    for _ in range(length):
        logits = model(context)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())


def main():
    # Load training data to build tokenizer
    data_path = "data/training.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build tokenizer
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model
    model = Transformer(vocab_size=vocab_size, d_model=128, num_layers=2, heads=4, ff_hidden_dim=256)
    model.load_state_dict(torch.load("checkpoints/transformer.pt", map_location=device))
    model.to(device)

    # Generate text
    result = sample(model, tokenizer, start_text="Hello wor", length=200, device=device)
    print("\nGenerated text:\n", result)


if __name__ == "__main__":
    main()
