
""" 
bpetokenizer.py

A basic implementation of a Byte Pair Encoding (BPE) tokenizer.
"""

import re
from collections import Counter

class BPETokenizer:
    
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = set()
        self.merges = []
        self.token2id = {}
        self.id2token = {}
    
    # Counts frequency of all adjacent symbol pairs (bigrams) in the vocabulary
    def get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                # Count how many times this pair occurs (weighted by word frequency)
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    # Performs one merge operation on the vocabulary by replacing all occurrences of a given pair with the merged token
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab


    
    