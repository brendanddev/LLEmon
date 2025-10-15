
""" 
hf_tokenize.py

"""

from src.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer

def main():
    # Create tokenizer
    tokenizer = HuggingFaceTokenizer()
    
    # Train tokenizer on text and save
    tokenizer.train("data/training.txt", vocab_size=30000)
    tokenizer.save("models/hf_tokenizer.json")
    
    # Reload tokenizer and test
    reloaded_tokenizer = HuggingFaceTokenizer("models/hf_tokenizer.json")
    sample = "Hello LLEmon!"
    encoded = reloaded_tokenizer.encode(sample)
    decoded = reloaded_tokenizer.decode(encoded)
    
    print(f"Sample: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab Size: {reloaded_tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
    
    