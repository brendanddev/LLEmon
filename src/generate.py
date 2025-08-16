
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

# Prompt for text generation
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
