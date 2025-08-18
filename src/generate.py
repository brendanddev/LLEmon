
""" 
generate.py 
A script to generate text using the fine tuned GPT-2 model.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from pathlib import Path 
import torch 


class Generator:
    
    def __init__(self, model_dir="models/final", seed=42, max_history_length=5):
        set_seed(seed)
        
        self.model_dir = Path(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.conversation_history = []
        self.MAX_HISTORY_LENGTH = max_history_length
        
        print(f"Model loaded from {self.model_dir} on {self.device}")
    
    
    def build_context(self, user_input: str) -> str:
        context = ""
        for exchange in self.conversation_history[-self.MAX_HISTORY_LENGTH:]:
            context += f"You: {exchange['user']}\nModel: {exchange['model']}\n"
        context += f"You: {user_input}\nModel:"
        return context
    
    def generate_response(self, context: str, max_new_tokens=100) -> str:
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_response = full_response.split("Model:")[-1].strip()
        
        if "You:" in model_response:
            model_response = model_response.split("You:")[0].strip()

        if not model_response:
            model_response = "I'm not sure how to respond to that."

        return model_response 

    def generate_text(self):
        print("Type 'quit', 'exit', or 'q' to stop generation.")
        print("Type 'clear' to clear conversation history.")
        print("-" * 50)

        while True:
            user_input = input("You: ")
            
            if user_input.lower().strip() in {"quit", "exit", "q"}:
                print("Exiting generation...")
                break
            
            if user_input.lower().strip() == "clear":
                self.conversation_history = []
                print("Conversation history cleared.")
                continue
            
            if not user_input.strip():
                continue
            
            context = self.build_context(user_input)
            model_response = self.generate_response(context)
            
            print(f"Model: {model_response}")
            
            self.conversation_history.append({
                "user": user_input,
                "model": model_response
            })