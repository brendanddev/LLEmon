
""" 
attention.py

Implements multi-head self-attention mechanism, allowing each token to 'look at' 
other tokens in the sequence for context.
"""

import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    
    def __init__(self, d_model, heads=8):
        super().__init__()
        
        # Ensure d_model is divisible by number of heads so each head has equal share of dimensions
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        
        # Linear layers for query, key, and value projections
        
        # Linear projections for query, key, and value
        # Each token embedding is transformed into three different views
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection to combine information from all heads
        self.out = nn.Linear(d_model, d_model)

    # Computes multi-head self-attention for a batch of sequences
    def forward(self, x):
        
        # B = batch size, T = sequence length, C = embedding dimension
        # H = number of heads, d_k = dimension per head
        B, T, C = x.shape
        H = self.heads
        d_k = self.d_k
        
        # Linear projectsion: Each token embedding is projected into query, key, and value vectors
        Q = self.q_linear(x).view(B, T, H, d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, T, H, d_k).transpose(1, 2)

        # Compute attention scores: Dot product between Q and K tells us how similar each token is to every other token
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply softmax to get attention weights that sum to 1 along the 'key' dimension
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values: Multiply attention weights by values (V) to get the new representation
        out = attn @ V
        
        # Concatenate heads and put through final linear layer
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear layer to mix information from all heads back into model dimension
        return self.out(out)