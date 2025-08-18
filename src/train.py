
""" 
train.py
Defines the Trainer class used to fine-tune a GPT-2 model on a custom text dataset.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer as HFTrainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset 
from pathlib import Path 
from utils.progress_callback import ProgressCallback
import torch


class Trainer:
    
    def __init__(self, dataset_path="data/training.txt", model_name="gpt2", output_dir="models"):
        """ Initializes the Trainer with dataset path, model name, and output directory """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add special tokens if they don't exist
        special_tokens = {"pad_token": "<|pad|>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Vocabulary size: {len(self.tokenizer)}")
    
    def load_dataset(self):
        """ Loads the dataset from the specified file path """
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Split by <|endoftext|> and clean up
        texts = content.split('<|endoftext|>')
        texts = [text.replace('<|startoftext|>', '').strip() for text in texts if text.strip()]
        
        print(f"Loaded {len(texts)} training examples")
        return Dataset.from_dict({"text": texts})

    def tokenize(self, batch):
        """ Tokenizes the input text batch with proper handling of special tokens """
        tokens = self.tokenizer(
            batch["text"], 
            truncation=True, 
            padding=False,
            max_length=512,
            return_tensors=None
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    def train(self, num_epochs=8, batch_size=4, learning_rate=5e-5):
        """ Trains the model on the loaded dataset """
        dataset = self.load_dataset()
        tokenized_dataset = dataset.map(self.tokenize, batched=True, remove_columns=["text"])
        
        # Use data collator for better batching
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=500,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="no",
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            report_to="none",
            load_best_model_at_end=False,
            logging_dir="../logs",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            seed=42,
        )

        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback] if hasattr(self, 'ProgressCallback') else None
        )

        print("Starting training...")
        trainer.train()

        final_dir = self.output_dir / "final"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        print(f"Model and tokenizer saved to {final_dir}")
        
        # Print training summary
        print(f"Training completed with {num_epochs} epochs")
        print(f"Final model saved to: {final_dir}")