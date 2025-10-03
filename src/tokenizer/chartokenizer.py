
""" 
chartokenizer.py

A simple character-level tokenizer.
"""

import torch

class CharTokenizer:
    
    def __init__(self, text):
        # Sorted set of unique characters in the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Mappings from char to index and index to char
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s):
        # Convert string to list of indices
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        # Convert list of indices back to string
        return "".join([self.itos[i] for i in l])