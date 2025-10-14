
""" 
huggingface_tokenizer.py

"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class HuggingFaceTokenizer:

    # Initializes the tokenizer, loading from vocab_path if provided
    def __init__(self, vocab_path):
        if vocab_path:
            self.tokenizer = Tokenizer.from_file(vocab_path)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
        self.vocab_size = self.tokenizer.get_vocab_size()
        
    def train(self, files, vocab_size=30000):
        
        
    def encode(self, text):
        pass
    
    def decode(self, tokens):
        pass