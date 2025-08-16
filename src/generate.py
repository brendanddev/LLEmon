
""" 
generate.py 
A script to generate text using the fine tuned GPT-2 model.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine tuned model 
tokenizer = GPT2Tokenizer.from_pretrained("models")
model = GPT2LMHeadModel.from_pretrained("models")

# Prompt for text generation
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(inputs["input_ids"], max_length=100, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
