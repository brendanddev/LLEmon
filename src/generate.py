
""" 
generate.py 
Defines the Generator class used to generate text using a fine-tuned GPT-2 model.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from pathlib import Path 
import torch 
import re


class Generator:
    
    def __init__(self, model_dir="models/final", seed=42, max_history_length=3):
        """ Initializes the text generator with the specified model directory """
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
        print(f"Vocabulary size: {len(self.tokenizer)}")
    
    def clean_response(self, response):
        """Clean up the generated response"""
        # Remove repetitive patterns
        response = re.sub(r'(.+?)\1{2,}', r'\1', response)
        
        # Stop at natural sentence endings
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) > 1:
            # Take the first complete sentence plus one more if it's substantial
            cleaned = sentences[0]
            if len(sentences) > 1 and len(sentences[1].strip()) > 10:
                cleaned += '. ' + sentences[1]
            cleaned = cleaned.strip() + '.'
        else:
            cleaned = response.strip()
        
        return cleaned
    
    def build_context(self, user_input: str, use_history: bool = True) -> str:
        """Build context for generation"""
        if not use_history or not self.conversation_history:
            return user_input
        
        context = ""
        for exchange in self.conversation_history[-self.MAX_HISTORY_LENGTH:]:
            context += f"{exchange['user']} {exchange['model']} "
        context += user_input
        return context.strip()
    
    def generate_response(self, context: str, max_new_tokens=80, temperature=0.7) -> str:
        """Generate response with improved parameters"""
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=15,
                do_sample=True,
                temperature=temperature,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if len(full_response) > len(context):
            model_response = full_response[len(context):].strip()
        else:
            model_response = "I need more context to provide a meaningful response."
        
        # Clean up the response
        model_response = self.clean_response(model_response)
        
        if not model_response or len(model_response) < 5:
            model_response = "Let me think about that differently."

        return model_response 

    def generate_single(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate a single response without conversation history"""
        context = prompt
        response = self.generate_response(context, max_new_tokens=max_tokens)
        return response

    def generate_text(self):
        """Interactive text generation with conversation"""
        print("Type 'quit', 'exit', or 'q' to stop generation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'single <prompt>' to generate without conversation context.")
        print("Type 'temp <value>' to change temperature (0.1-1.5).")
        print("-" * 60)
        
        current_temp = 0.7

        while True:
            user_input = input("You: ")
            
            if user_input.lower().strip() in {"quit", "exit", "q"}:
                print("Exiting generation...")
                break
            
            if user_input.lower().strip() == "clear":
                self.conversation_history = []
                print("Conversation history cleared.")
                continue
                
            if user_input.lower().startswith("temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 1.5:
                        current_temp = new_temp
                        print(f"Temperature set to {current_temp}")
                    else:
                        print("Temperature should be between 0.1 and 1.5")
                except:
                    print("Invalid temperature value")
                continue
                
            if user_input.lower().startswith("single "):
                prompt = user_input[7:]
                if prompt.strip():
                    response = self.generate_single(prompt, max_tokens=120)
                    print(f"Model: {response}")
                continue
            
            if not user_input.strip():
                continue
            
            # Use conversation context
            context = self.build_context(user_input)
            model_response = self.generate_response(context, temperature=current_temp)
            
            print(f"Model: {model_response}")
            
            # Add to history
            self.conversation_history.append({
                "user": user_input,
                "model": model_response
            })
            
            # Keep history manageable
            if len(self.conversation_history) > self.MAX_HISTORY_LENGTH:
                self.conversation_history.pop(0)