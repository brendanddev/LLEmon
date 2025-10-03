
""" 
positional_encoding.py

Injects information about token positions using sine and cosine functions.
This lets the Transformer model know the order of tokens in a sequence,
since it has no reccurrent or convolution built in.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create matrix of shape (max_len, d_model) to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Position indices
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Compute frequency terms for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Apply sin to even dimensions and cos to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension so it can be broadcast with embeddings
        pe = pe.unsqueeze(0)
        
        # Save as a buffer so its not a model parameter
        self.register_buffer('pe', pe)
    
    def forward():
