

import torch
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # Batch size, sequence length, vocabulary size
x = torch.randn(B, T, C)
x.shape  # Should be (B, T, C)
torch.Size([B, T, C])  # Should be [4, 8, 2]
print(x.shape)  # Should print torch.Size([4, 8, 2])

# Bag-of-words (bow) representation
# Initialize an empty tensor to hold the bag-of-words representation
xbow = torch.zeros((B, T, C))
# Iterate over batch dimensions
for b in range(B):
    # Time
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, dim=0)  # Compute the mean over the sequence length

print(xbow.shape)  # Should print torch.Size([4, 8, 2])
print(xbow)  # Should print the computed xbow tensor