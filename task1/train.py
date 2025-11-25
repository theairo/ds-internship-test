import torch
import json
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification,
    pipeline
)
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Label Mappings
labels = ["O", "B-MNT", "I-MNT"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_and_align_labels(examples):
    """
    Tokenizes and aligns word-level NER labels for training a token classification model.
    
    This function:
    1. Tokenizes the words
    2. Aligns BIO labels to the new tokens.
    3. Converts 'B-' to 'I-' for any subsequent subword pieces
    4. Sets the special tokens to -100 so they are ignored in loss computation.

    Args:
        examples (dict): {"tokens": [...], "ner_tags": [...]} for a batch of examples.

    Returns:
        dict: tokenized inputs with an added "labels" key aligned for training.
    """
    # Tokenize the words
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        is_split_into_words=True, 
        truncation=True
    )
    
    labels_aligned = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignored in loss
            elif word_idx != previous_word_idx:
                # Start of a new word
                label_ids.append(label2id[label[word_idx]])
            else:
                # Subword parts: convert B- tag to I- tag
                if label[word_idx].startswith("B-"):
                    label_ids.append(label2id[label[word_idx].replace("B-", "I-")])
                else:
                    label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels_aligned.append(label_ids)
        
    tokenized_inputs["labels"] = labels_aligned
    return tokenized_inputs

# Load Dataset from the JSONL file
print("Initializing data loading...")

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / "data" / "final"

# Define the files
data_files = {
    "train": str(data_dir / "train.jsonl"),
    "validation": str(data_dir / "validation.jsonl"), 
    "test": str(data_dir / "test.jsonl")
}

dataset_hf = load_dataset("json", data_files=data_files)
tokenized_datasets = dataset_hf.map(tokenize_and_align_labels, batched=True)

# Model Initialization
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Data Collator (dynamic padding, tensor conversion, attention mask)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Training Arguments (default for NER)
training_args = TrainingArguments(
    output_dir="./mnt_ner_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    num_train_epochs=5, 
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("./mnt_ner_model")
tokenizer.save_pretrained("./mnt_ner_model")