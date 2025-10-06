
""" 
chartokenizer.py

A simple character-level tokenizer implementation.
This tokenizer converts characters to integers and vice versa.
"""

class CharTokenizer:
    
    def __init__(self, text):
        chars = sorted(list(set(text)))
        
        # Maps each character to a unique integer
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        
        # Maps each integer back to its character
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        self.vocab_size = len(chars)
    
    # Converts a string of text into a list of integer tokens
    def encode(self, text):
        return [self.stoi[c] for c in text]

    # Converts a list of integer tokens back into a string of text
    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])