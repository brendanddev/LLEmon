
""" 
transformer.py

Defines the overall Transformer model built from stacked TransformerBlocks.
"""

import torch.nn as nn
from .positionalencoding import PositionalEncoding
from .transformerblock import TransformerBlock

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, heads=4, ff_hidden_dim=512):
        super().__init__()
        
        # Convert integer token IDs into dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Add position information so model knows token order
        self.pos_encoding = PositionalEncoding(d_model)

        # Stack multiple TransformerBlocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])

        # Final normalization before output
        self.ln = nn.LayerNorm(d_model)
        
        # Linear projection back to vocabulary size (for logits / predictions)
        self.fc_out = nn.Linear(d_model, vocab_size)

    # Forward pass through the entire Transformer model
    def forward(self, x):
        
        # Token + positional embeddings (Shape: [B, T, d_model])
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Normalize and project to output vocabulary
        x = self.ln(x)
        return self.fc_out(x)
    