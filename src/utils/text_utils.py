
""" 
text_utils.py 
Utility functions for cleaning and processing text data

Brendan Dileo, August 2025
"""

import re

def remove_wiki_refs(text):
    """ Removes bracketed numeric references from the text """
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    return cleaned_text 

def remove_multiple_whitespace(text):
    """ Removes multiple whitespace characters and trims the text """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text):
    """ Cleans the text by removing wiki references and multiple whitespace """
    text = remove_wiki_refs(text)
    text = remove_multiple_whitespace(text)
    return text

if __name__ == "__main__":
    print("Paste your text below. Finish with an empty line to end:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    
    input_text = "\n".join(lines)
    cleaned = clean_text(input_text)
    
    print("\nCleaned text:\n")
    print(cleaned)