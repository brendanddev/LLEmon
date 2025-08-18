
""" 
data_utils.py 
Utility functions for analyzing a provided text dataset.

Brendan Dileo, August 2025
"""

from collections import Counter

def count_lines(texts):
    """ Counts and returns the number of non empty lines in the texts """
    return len([line for line in texts if line.strip()])

def count_chars(texts):
    """ Counts and returns the total number of characters in the texts """
    return sum(len(line) for line in texts)

def count_unique_chars(texts):
    """ Counts and returns the number of unique characters in the texts """
    chars = set()
    for line in texts:
        chars.update(line.strip())
    return len(chars)
    
def count_char_frequency(texts):
    """ Counts and returns the frequency of each character in the texts """
    counter = Counter()
    for line in texts:
        counter.update(line.strip())
    return counter

def load_dataset(file_path):
    """ Loads a text dataset from a file and returns a list of non-empty lines """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts

def run_analysis(file_path="data/training.txt"):
    texts = load_dataset(file_path)
    print(f"Lines: {count_lines(texts)}")
    print(f"Characters: {count_chars(texts)}")
    print(f"Unique characters: {count_unique_chars(texts)}")
    print("Top 10 most common characters:", count_char_frequency(texts).most_common(10))
    return texts