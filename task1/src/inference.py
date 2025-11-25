import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from pathlib import Path
import pandas as pd

class MountainNER:
    def __init__(self, device=None):

        # If doesn't specify, auto-detect CUDA
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model is stored on Hugging Face
        repo_id = "nikolai-domashenko/mnt-ner-model"

        print(f"Loading model from {repo_id} to {self.device}...")

        # Load Artifacts
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForTokenClassification.from_pretrained(repo_id)
        
        # Move model to GPU if available
        self.model.to(self.device)
        self.model.eval() # Important: Set to evaluation mode (turns off Dropout)

        # Load Labels dynamically from the model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict(self, sentences):
        # Handle single string vs list input
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Tokenize (using self.tokenizer)
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            is_split_into_words=False
        ).to(self.device) # Move data to GPU if available
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get Predictions
        # Move back to CPU for numpy, calculate argmax
        predictions = outputs.logits.argmax(dim=2).cpu().numpy()
        
        results = []

        # The Alignment Logic (Your Logic + Fixes)
        for i, sentence in enumerate(sentences):
            word_ids = inputs.word_ids(batch_index=i)
            previous_word_idx = None
            extracted_entities = []
            
            # Get the tokens so we don't rely on .split()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])

            current_word = ""
            current_label = None

            for idx, word_idx in enumerate(word_ids):
                # Skip special tokens (None)
                if word_idx is None:
                    continue
                
                # If it's a new word (start of a word)
                if word_idx != previous_word_idx:
                    # Save the previous word if it existed
                    if current_word:
                        extracted_entities.append((current_word, current_label))
                    
                    # Start a new word
                    current_word = tokens[idx]
                    current_label = self.id2label[predictions[i][idx]]
                    
                else:
                    # It's a subword, append it to current_word
                    part = tokens[idx]
                    if part.startswith("##"):
                        current_word += part[2:]
                    else:
                        current_word += part
                        
                previous_word_idx = word_idx
            
            # Append the last word
            if current_word:
                extracted_entities.append((current_word, current_label))
                
            results.append(extracted_entities)
        
        return results
    
    def clean_predictions(self, batch_results):
        """
        Cleans the raw BIO tags into human-readable strings.
        Handles the batch output from predict().
        """
        batch_cleaned = []

        # Loop through each sentence in the batch
        for sentence_results in batch_results:
            clean_entities = []
            current_entity = []
            
            for word, label in sentence_results:
                # B-MNT: Start of a new entity
                if label.startswith("B-"):
                    if current_entity:
                        clean_entities.append(" ".join(current_entity))
                        current_entity = []
                    current_entity.append(word)
                
                # I-MNT: Continuation of an entity
                elif label.startswith("I-") and current_entity:
                    current_entity.append(word)
                
                # O: Outside or non-mountain
                else:
                    if current_entity:
                        clean_entities.append(" ".join(current_entity))
                        current_entity = []
            
            # Catch any entity left at the very end of the sentence
            if current_entity:
                clean_entities.append(" ".join(current_entity))
            
            batch_cleaned.append(clean_entities)
            
        return batch_cleaned
    
    def evaluate_file(self, test_file_path):
        """
        Loads a test dataset, calculates Overall metrics, 
        and then splits metrics by Language (EN/UA).
        """
        import json
        from seqeval.metrics import classification_report, f1_score
        
        print(f"Loading test data from {test_file_path}...")
        
        # Lists to store data
        sentences = []
        ground_truth = []
        languages = []
        
        # Load File
        with open(test_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f]

        # Extract data
        for item in data:
            sentences.append(" ".join(item['tokens']))
            ground_truth.append(item['ner_tags'])
            languages.append(item.get('language', 'unknown'))

        # Run Prediction
        print(f"Running inference on {len(sentences)} examples...")
        batch_predictions = self.predict(sentences)
        
        # Align Predictions with Ground Truth
        true_labels_all = []
        pred_labels_all = []
        
        for i, sent_preds in enumerate(batch_predictions):
            preds_only = [label for word, label in sent_preds]
            true_only = ground_truth[i]
            
            # Safety Truncation
            min_len = min(len(preds_only), len(true_only))
            pred_labels_all.append(preds_only[:min_len])
            true_labels_all.append(true_only[:min_len])
        metrics_report = {}

        # Calculate Overall Metrics
        metrics_report['Overall'] = self._get_filtered_report(true_labels_all, pred_labels_all)

        # Calculate Per-Language Metrics
        unique_langs = set(languages)
        for lang in unique_langs:
            lang_true = [t for t, l in zip(true_labels_all, languages) if l == lang]
            lang_pred = [p for p, l in zip(pred_labels_all, languages) if l == lang]
            
            if lang_true:
                metrics_report[lang] = self._get_filtered_report(lang_true, lang_pred)

        return metrics_report

    def _get_filtered_report(self, true_labels, pred_labels):
        """
        Helper: Generates report and removes micro/macro/weighted averages.
        """
        from seqeval.metrics import classification_report
        
        # Generate the full dictionary
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        # Remove the clutter
        # .pop(key, None) safely removes the key if it exists, without crashing if it's missing
        for key in ['micro avg', 'macro avg', 'weighted avg']:
            report.pop(key, None)
            
        return report

# Command line interface (CLI)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NER Inference & Evaluation")

    parser.add_argument("--test_file", type=str, default="./data/final/test.jsonl", help="Path to test dataset")

    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict", nargs="+", help="Run prediction on given sentences")
    
    args = parser.parse_args()

    # Loads the heavy weights into memory
    try:
        ner_system = MountainNER()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # handle prediction
    if args.predict:
        print(f"\nRunning inference on {len(args.predict)} sentence(s)...\n")
        
        # Call the method inside the class
        results = ner_system.predict(args.predict)
        
        
        # The class returns raw data, so we print it nicely here in the CLI
        for sentence, entities in zip(args.predict, results):
            print(f"Input: {sentence}")
            print(f"Found: {entities}")
            print("-" * 30)
        
        print("\n", ner_system.clean_predictions(results))
    
    # handle evaluation
    if args.test:
        # Run evaluation (Returns a dictionary of dictionaries)
        results = ner_system.evaluate_file(args.test_file)

        # Loop through each report (Overall, EN, UA) and print nicely
        for language, metrics_dict in results.items():
            print(f"\n{'='*20}")
            print(f"REPORT: {language}")
            print(f"{'='*20}")
            
            # Convert to DataFrame for pretty printing
            df = pd.DataFrame(metrics_dict).transpose()
            
            df_clean = df.drop(['micro avg', 'macro avg', 'weighted avg'], errors='ignore')
            
            # Round numbers to 2 decimal places for readability
            print(df_clean.round(2).to_string())

