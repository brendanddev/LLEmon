
""" 
generate.py 
A script to generate text using the fine tuned GPT-2 model.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from pathlib import Path 
import torch 

set_seed(42)

# Project paths
project_root = Path(__file__).parent.parent
model_dir = project_root / "models" / "final"

# Load fine tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}")
print("Type 'quit', 'exit', or 'q' to stop generation.")
print("Type 'clear' to clear conversation history.")
print("-" * 50)

# Store conversation history
conversation_history = []
MAX_HISTORY_LENGTH = 5  

## Generate text using the model
while True:
    user_input = input("You: ")
    
    if user_input.lower().strip() in {"quit", "exit", "q"}:
        print("Exiting generation...")
        break
    
    if user_input.lower().strip() == "clear":
        conversation_history = []
        print("Conversation history cleared.")
        continue
    
    if not user_input.strip():
        continue
    
    # Build conversation context from recent history
    context = ""
    for exchange in conversation_history[-MAX_HISTORY_LENGTH:]:
        context += f"You: {exchange['user']}\nModel: {exchange['model']}\n"
    
    # Add current user input
    context += f"You: {user_input}\nModel:"
    
    # Encode input
    inputs = tokenizer(
        context, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the models response
    try:
        model_response = full_response.split("Model:")[-1].strip()
        
        # Clean up the response
        if "You:" in model_response:
            model_response = model_response.split("You:")[0].strip()
        
        # Ensure response isn't empty
        if not model_response:
            model_response = "I'm not sure how to respond to that."
        
        print(f"Model: {model_response}")
        
        # Add to conversation history
        conversation_history.append({
            "user": user_input,
            "model": model_response
        })
        
    except Exception as e:
        print(f"Error processing response: {e}")
        print("Model: Sorry, I encountered an error generating a response.")