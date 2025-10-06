
""" 
transformerblock.py

Defines a single Transformer block, which combines multi-head self-attention, a feed forward network,
and layer normalization with residual connections.
"""

import torch.nn as nn
from .attention import Attention
from .feedforward import FeedForward

class TransformerBlock:
    
    def __init__(self, d_model, heads, ff_hidden_dim):
        super().__init__()
        
        # Multi-head self-attention sublayer
        self.attn = Attention(d_model, heads)
        
        # Feed-forward sublayer (applied position-wise)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        
        # Layer normalization for stabilizing training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    # Forward pass for one Transformer block
    def forward(self, x):
        # Attention + residual
        
        # Self attention + residual connection
        # Normalize first, add result of attention back to input
        x = x + self.attn(self.norm1(x))
        
        # Feedforward + residual connection
        # Normalize again before passing through feedforward sublayer
        # Add output back to input
        x = x + self.ff(self.norm2(x))
        return x
