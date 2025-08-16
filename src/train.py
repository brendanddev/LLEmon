
""" 
train.py
A script to fine tune a GPT-2 model on a custom text dataset.

Brendan Dileo, August 2025
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset 

# Load text dataset
with open("./data/", "r", encoding="utf-8") as file:
    texts = [line.strip() for line in file if line.strip()]

# Create a dataset from the texts
dataset = Dataset.from_dict({"text": texts})

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the data
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="../models",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training the model
trainer.train()