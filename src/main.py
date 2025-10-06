
""" 
main.py 
Main entry point for the LLEmon text generation system.

Brendan Dileo, August 2025
"""

import torch
from tokenizer.chartokenizer import CharTokenizer
from models.transformerblock import TransformerBlock
from models.transformer import Transformer

def main():
    text = "hello world"
    tokenizer = CharTokenizer(text)
    
    # Tokenize text
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    
    # Convert to tensor
    x = torch.tensor(encoded).unsqueeze(0)
    
    # Initialize transformer model
    vocab_size = tokenizer.vocab_size
    model = Transformer(vocab_size=vocab_size, d_model=64, num_layers=2, heads=4, ff_hidden_dim=128)
    
    # Forward pass
    out = model(x)
    print("Model output shape:", out.shape)
    
    # Predict next token
    probs = torch.softmax(out, dim=-1)
    next_token_id = torch.argmax(probs[:, -1, :], dim=-1).item()
    next_token = tokenizer.decode([next_token_id])
    print("Next token prediction:", next_token)
    


if __name__ == "__main__":
    main()