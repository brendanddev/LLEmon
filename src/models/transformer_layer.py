
""" 
transformer_layer.py

Defines a single Transformer encoder layer, combining multi-head self attention, add & norm,
and a feedforward network.
"""

import torch.nn as nn
from .attention import Attention
from .feedforward import FeedForward

class TransformerLayer(nn.Module):
    
    def __init__(self, d_model, heads, d_ff=2048):
        super().__init__()
        self.attn = Attention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self attention block
        x2 = self.attn(x, x, x, mask)
        x = self.norm1(x + x2)
        
        # Feedforward block
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x