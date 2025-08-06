
""" 
char_tokenizer.py
Defines a character level tokenizer for text processing

Brendan Dileo, August 2025
"""

from src.utils.text_processing import load_text, count_characters


class CharTokenizer:
    
    def __init__(self, text):
        """ 
        Initializes the CharTokenizer with a given text
        
        """
        self.chars = sorted(set(text))
        # Creates character to index and index to character mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        """ Encodes a text string into a list of token indices """
        return [self.stoi[ch] for ch in text if ch in self.stoi]
        
    def decode(self, tokens):
        """ Decodes a list of token indices back into a text string """
        return ''.join(self.itos[token] for token in tokens if token in self.itos)
