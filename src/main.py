
""" 
main.py 
Main entry point for the LLEmon text generation system.

Brendan Dileo, August 2025
"""

import sys
from pathlib import Path
import train
import generate
from utils import data_utils, token_utils

sys.path.append(str(Path(__file__).parent.parent))


def main():
    """ Main function to run the app """
    print("Welcome to the LLEmon text generation system!")
    print("You can train a model, generate text, or analyze data.")
    
    while True:
        print("Options:")
        print("1. Train a model")
        print("2. Generate text using a trained model")
        print("3. Analyze text data")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            print("Starting training...")
            train.train_model()
        elif choice == '2':
            print("Starting text generation...")
            generate.generate_text()
        elif choice == '3':
            print("Analyzing text data...")
            texts = data_utils.load_dataset("data/training.txt")
            token_utils.run_token_analysis("data/training.txt")
        elif choice == '4':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
