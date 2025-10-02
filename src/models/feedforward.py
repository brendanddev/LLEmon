
""" 
feedforward.py

Implements a basic feedforward network compoment of transformer architectures.
After the attention mechanism processes relationships between tokens, the feedforward network
processes each position independently to transform the representations.
"""

import torch.nn as nn

class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        
        # First linear transformation: expand from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation: contract from d_ff back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.linear1(x).relu())