
""" 
positionalencoding.py

Implements sinusoidal positional encoding.
These encoding allow the model to understand where each token is in the sequence.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a tensor to hold positional encodings (Shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector representing the position of each token in the sequence
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Computes a scaling term for each dimension of the embedding
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it can be added to input embeddings
        pe = pe.unsqueeze(0)
        
        # Register as a non-learnable buffer
        self.register_buffer('pe', pe)
    
    # Adds positional encoding to input embeddings
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x