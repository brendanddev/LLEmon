
""" 
main.py 
Main entry point for the LLEmon text generation system.

Brendan Dileo, August 2025
"""

from models.transformer import Transformer
from tokenizer.bpetokenizer import BPETokenizer
import torch

def main():
    
    text = "Hello, world!"
    
    # Build tokenizer
    tokenizer = BPETokenizer(num_merges=100)
    tokenizer.fit(text)
    
    # Encode text into token IDs
    ids = tokenizer.encode(text)
    
    # Convert to tensor with batch dimension
    x = torch.tensor([ids], dtype=torch.long)
    
    # Initialize model
    model = Transformer(vocab_size=len(tokenizer.vocab))
    
    # Forward pass
    out = model(x)
    
    print(out.shape)  # Should be (1, sequence_length, vocab_size)

if __name__ == "__main__":
    main()
