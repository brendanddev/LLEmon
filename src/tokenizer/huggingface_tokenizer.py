
""" 
huggingface_tokenizer.py

A lightweight wrapper around Hugging Face's tokenizers library for training, saving, loading, 
and using a BPE tokenizer.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

class HuggingFaceTokenizer:

    # Initializes the tokenizer, loading from vocab_path if provided
    def __init__(self, vocab_path=None):
        if vocab_path:
            self.tokenizer = Tokenizer.from_file(vocab_path)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    # Trains the tokenizer on one or more text files
    def train(self, files, vocab_size=30000):
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
        )
        self.tokenizer.train(files=files, trainer=trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
    # Encodes text into a list of token IDs
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    # Decodes a list of token IDs back into text
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    # Saves the tokenizer to a file
    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.tokenizer.save(path)
        self.vocab_path = path