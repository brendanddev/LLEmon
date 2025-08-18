
""" 
train.py
A script to fine tune a GPT-2 model on a custom text dataset.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer as HFTrainer, TrainingArguments
from datasets import Dataset 
from pathlib import Path 


class Trainer:
    
    def __init__(self, dataset_path="data/training.txt", model_name="gpt2", output_dir="models"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            texts = [line.strip() for line in file if line.strip()]
        return Dataset.from_dict({"text": texts})

    def tokenize(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    def train(self, num_epochs=1, batch_size=2):
        dataset = self.load_dataset()
        tokenized_dataset = dataset.map(self.tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=False,
            logging_dir="../logs",
        )

        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()

        final_dir = self.output_dir / "final"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        print(f"Model and tokenizer saved to {final_dir}")