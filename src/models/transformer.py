
""" 
transformer.py

Defines a basic implementation of the Transformer model architecture.
Combines multiple TransformerLayer blocks, adds embeddings for tokens, adds positional encodings,
and projects to output vocabulary size for language modeling tasks.
"""

import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .transformer_layer import TransformerLayer

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model=128, N=2, heads=8, d_ff=512, max_len=100):
        super().__init__()
        
        # Token embedding layer: convert word indices to dense vectors
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Add positional information to embeddings
        self.pos = PositionalEncoding(d_model, max_len)
        
        # Stack of N Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, heads, d_ff) for _ in range(N)])
        
        # Final linear layer maps hidden states to vocab logits
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos(x)
        
        # Pass through stacked Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Project hidden states back to vocab size
        return self.fc_out(x)