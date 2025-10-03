
""" 
bpetokenizer.py

A basic implementation of a Byte Pair Encoding (BPE) tokenizer.
"""

import re
from collections import Counter, defaultdict

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
        
        # Regex to match the pair as a full unit (not part of a larger token)
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            # Replace all instances of the bigram with the merged token
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    # Trains the tokenizer on input text by performing a series of merges
    def fit(self, text):
        # Build base vocab from words with end marker
        words = text.strip().split()
        vocab = defaultdict(int)
        for w in words:
            # Add end of word token
            vocab[' '.join(list(w)) + " </w>"] += 1

        # Perform byte pair encoding merges
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)
        
        # Collect final tokens in vocabulary
        self.vocab = set()
        for word in vocab:
            self.vocab.update(word.split())
        
        # Build id mappings
        self.token2id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
    
    # Tokenizes a single word into subword tokens using the learned merges
    def tokenize(self, word):
        chars = list(word) + ['</w>']
        i = 0
        while i < len(chars) - 1:
            pair = (chars[i], chars[i+1])
            if pair in self.merges:
                # Merge if this pair was learned while training
                chars[i:i+2] = [''.join(pair)]
            else:
                i += 1
        # Drop the end of word token for output
        return chars[:-1]