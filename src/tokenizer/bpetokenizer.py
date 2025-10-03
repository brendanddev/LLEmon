
""" 
bpetokenizer.py

A basic implementation of a Byte Pair Encoding (BPE) tokenizer.
"""

import re
from collections import defaultdict, Counter

class BPETokenizer:
    
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = set()
        self.merges = []
        self.token2id = {}
        self.id2token = {}
        
    def get_stats(self, vocab):
        pass
        
    def merge():
        pass
    
    def fit(self, text):
        pass