
""" 
feedforward.py

Implements the position-wise feed forward network used inside each Transformer layer.
After self-attention mixes information across tokens, this FFN applies a small neural network independently
to each tokens representation to further process it.

Equation from the Transformer paper ("Attention Is All You Need"):
    FFN(x) = max(0, xW1 + b1)W2 + b2
"""

import torch.nn as nn

class FeedForward(nn.Module):
    
    def __init__(self, d_model, hidden_dim=2048):
        super().__init__()
        
        # Define the feedforward network as a simple stack of layers
        # Linear layer expands demonsionality from d_model to hidden_dim
        # ReLU activation introduces non-linearity
        # Linear layer projects back down to d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
    
    # Applies the feedforward network to each token's embedding
    def forward(self, x):
        return self.net(x)
    