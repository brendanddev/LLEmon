
"""
sample.py
Generates text from a trained Transformer model.

Run: python -m src.utils.sample
"""

import torch
from src.models.transformer import Transformer
from src.tokenizer.chartokenizer import CharTokenizer
from src.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer

def sample(model, tokenizer, start_text="Hello", length=100, device="cpu"):
    model.eval()
    
    # Encode starting text
    context = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(length):
            logits = model(context)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    
    return tokenizer.decode(context[0].tolist())

def main():
    tokenizer_path = "models/hf_tokenizer.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = HuggingFaceTokenizer(vocab_path=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Load trained model
    model = Transformer(vocab_size=vocab_size, d_model=256, num_layers=4, heads=8, ff_hidden_dim=512)
    model.load_state_dict(torch.load("checkpoints/transformer.pt", map_location=device))
    model.to(device)
    
    # Generate text
    result = sample(model, tokenizer, start_text="Hello ", length=200, device=device)
    print("\nGenerated text:\n", result)

if __name__ == "__main__":
    main()