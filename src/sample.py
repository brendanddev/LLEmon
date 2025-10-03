
""" 
sample.py

Generates text samples from a trained Transformer model.
"""

import torch
from models.transformer import Transformer
from tokenizer.chartokenizer import CharTokenizer

def generate(model, tokenizer, prompt="Hello", length=100, device="cpu"):
    model.eval()
    idxs = tokenizer.encode(prompt)
    x = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(length):
        logits = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    
    return tokenizer.decode(x[0].tolist())

