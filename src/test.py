
""" 
test.py

Simple test script to test individual components of the LLEmon system.
"""

from tokenizer.chartokenizer import CharTokenizer
from models.transformer import Transformer

def test_tokenizer():
    sample_text = "hello world"

    # Create the tokenizer
    tokenizer = CharTokenizer(sample_text)

    # Encode the text
    encoded = tokenizer.encode(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Encoded tokens: {encoded}")

    # Decode the tokens back to text
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: {decoded}")

    # Show vocab details
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"stoi (char → id): {tokenizer.stoi}")
    print(f"itos (id → char): {tokenizer.itos}")

if __name__ == "__main__":
    test_tokenizer()