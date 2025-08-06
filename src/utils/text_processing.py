
""" 
text_processing.py
Defines text processing related utilities

Brendan Dileo, August 2025
"""

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
    