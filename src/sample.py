
""" 
sample.py

Generates text samples from a trained Transformer model.
"""

import torch
from models.transformer import Transformer
from tokenizer.bpetokenizer import BPETokenizer
import rust_bpe_tokenizer as rbt

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

def sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load("checkpoints/model.pth", map_location=device)

    # Reconstruct tokenizer from checkpoint
    tokenizer = BPETokenizer(num_merges=1000)
    tokenizer.vocab = set(checkpoint["vocab"])
    
    # Rebuild ID mappings
    tokenizer.token2id = {tok:i for i, tok in enumerate(sorted(tokenizer.vocab))}
    tokenizer.id2token = {i:tok for tok,i in tokenizer.token2id.items()}
    
    # Initialize model and load state
    model = Transformer(
        vocab_size=len(tokenizer.vocab),
        d_model=128, N=2, heads=8, d_ff=512, max_len=256
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    
    # Generate sample text
    text = generate(model, tokenizer, prompt="Once upon a time", length=200, device=device)
    print(text)

if __name__ == "__main__":
    sample()