
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
