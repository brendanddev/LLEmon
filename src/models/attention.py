
""" 
attention.py

Multi-head attention mechanism allowing a model to focus on different parts of the input sequence.
"""

import torch 
import torch.nn as nn
import math

class Attention(nn.Module):
    
    def __init__(self, d_model, heads=8):
        super().__init__()
        
        # Store model dimensions
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        
        # Linear transformations for Query, Key, and Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Final output projection to combine all heads
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # Get batch size for reshaping operations
        B = q.size(0)
        
        # Apply linear transformations and reshape for multi-head attention
        # Transform inputs through learned projections and split into multiple heads
        q = self.q_linear(q).view(B, -1, self.heads, self.d_k)
        k = self.k_linear(k).view(B, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(B, -1, self.heads, self.d_k)

        # Transpose to get shape [batch_size, heads, seq_len, d_k]
        # Allows us to process all heads in parallel
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        
        # Compute attention scores using scaled dot-product
        # scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attn, v)
        
        # Concatenate heads and reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # Apply final linear transformation
        return self.out(context)