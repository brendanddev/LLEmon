
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