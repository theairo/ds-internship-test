import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from pathlib import Path
import evaluate

# Paths
script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / "data" / "final"
model_dir = "./mnt_ner_model"

# Label Mappings
labels = ["O", "B-MNT", "I-MNT"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

metric = evaluate.load("seqeval")

# Metrics computation
def compute_metrics(pred_and_labels):
    preds, true_labels = pred_and_labels
    preds = np.argmax(preds, axis=2)

    true_preds, true_refs = [], []
    for pred_seq, label_seq in zip(preds, true_labels):
        filtered_preds, filtered_labels = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            filtered_preds.append(id2label[p])
            filtered_labels.append(id2label[l])
        true_preds.append(filtered_preds)
        true_refs.append(filtered_labels)

    results = metric.compute(predictions=true_preds, references=true_refs)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# Tokenizer and label alignment
tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
    # examples["ner_tags"] is a list of lists of strings (e.g., [["B-MNT", "O"], ...])
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

# Load the model and trainer
model = AutoModelForTokenClassification.from_pretrained(model_dir)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="/tmp_eval", per_device_eval_batch_size=8),
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

def evaluate_test_set():
    data_files = {"test": str(data_dir / "test.jsonl")}
    dataset = load_dataset("json", data_files=data_files)
    tokenized_test = dataset["test"].map(tokenize_and_align_labels, batched=True)

    print("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized_test)

    print("\n---------------- REPORT ----------------")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall:    {metrics['eval_recall']:.4f}")
    print(f"F1 Score:  {metrics['eval_f1']:.4f}")
    print(f"Loss:      {metrics['eval_loss']:.4f}")
    print("----------------------------------------")

def predict_sentences(sentences):
    inputs = tokenizer(sentences, is_split_into_words=False, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)

    for i, sentence in enumerate(sentences):
        word_ids = inputs.word_ids(batch_index=i) 
        previous_word_idx = None
        pred_labels = []

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                pred_labels.append(id2label[predictions[i][token_idx]])
            previous_word_idx = word_idx


        words = sentence.split()

        words = words[:len(pred_labels)]
        print(f"\nSentence: {sentence}")
        print("Predictions:", list(zip(words, pred_labels)))


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Inference & Evaluation")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict", nargs="+", help="Run prediction on given sentences")
    args = parser.parse_args()

    if args.test:
        evaluate_test_set()

    if args.predict:
        predict_sentences(args.predict)
