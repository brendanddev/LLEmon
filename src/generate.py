
""" 
generate.py 
A script to generate text using the fine tuned GPT-2 model.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

# Project paths
project_root = Path(__file__).parent.parent
model_dir = project_root / "models" / "final"

# Load fine tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

## Generate text using the model

while True:
    prompt = input("You: ")
    if prompt.lower().strip() in {"quit", "exit", "q"}:
        print("Exiting generation...")
        break
    
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    
    # Decode and print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model: {response}\n")