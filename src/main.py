
""" 
main.py 
Main entry point for the LLEmon text generation system.

Brendan Dileo, August 2025
"""

import rust_bpe_tokenizer
from tokenizer.bpetokenizer import BPETokenizer

def main():
    with open("data/training.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text_sample = text[:5000]

    print("Loaded text length:", len(text_sample))

    # Initialize tokenizer
    tokenizer = BPETokenizer(num_merges=50)
    tokenizer.fit(text_sample)
    
    print("Fitting done.")
    print("Vocabulary size:", len(tokenizer.vocab))
    
    prompt = "Once upon a time"
    encoded = tokenizer.encode(prompt)
    print("Encoded prompt:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded prompt:", decoded)


def tokenizer_test():

    # Create and train tokenizer
    tokenizer = rust_bpe_tokenizer.BpeTokenizer(50)
    tokenizer.fit("hello world hello byte pair encoding")
    
    # Test encoding
    ids = tokenizer.encode("hello world")
    print(f"Token IDs: {ids}")

    # Test decoding
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")

    # Check vocab size
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    

if __name__ == "__main__":
    tokenizer_test()