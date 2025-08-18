
""" 
train.py
A script to fine tune a GPT-2 model on a custom text dataset.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset 
from pathlib import Path


def train_model():
    # Project paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "models"
    
    # Load text dataset
    with open("data/training.txt", "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file if line.strip()]

    # Create a dataset from the texts
    dataset = Dataset.from_dict({"text": texts})

    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # Tokenize the data
    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        logging_dir="../logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training the model
    trainer.train()

    # Save final model & tokenizer
    final_dir = output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)