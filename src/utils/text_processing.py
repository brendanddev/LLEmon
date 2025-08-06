
""" 
text_processing.py
Defines text processing related utilities

Brendan Dileo, August 2025
"""

import re

def load_text(file_path):
    """ Load text from a file and return its content """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def clean_text(text):
    """ Clean the text by removing HTML tags, extra whitespace, emails, and URLs """
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'http\S+|www\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
    return cleaned_text.strip()