## Multilingual Mountain Named Entity Recognition
This repository contains NER system capable of identifying mountain names in both English and Ukrainian text.

The project demonstrates full machine learning pipeline, starting from synthetic dataset generation using LLM API and finishing with a fine-tuend BERT model wrapped in a class.

### Project Overview
The goal of this task was to identify a specifc entity type (mountains) in multilingual text. The challenge lies not only in training a model, but dealing with context ambiguity. For example, distuingishing between "Карпати" (mountain range) and "Карпати" (the football club).

### Solution Approach and Design Decisions
#### 1. Data Strategy: Synthetic Generation with LLMs
The problem was the lack of pre-existing, high-quality dataset for mountain names in Ukrainian (there are some datasets for English present). To solve manual labeling (or writing complex scraping algorithms) I built a data generation pipeline using the Gemini 2.5 Flash API (due to it's high limits for cost-free inference capabilities)
##### Generation and quantities
The generation was based on 4 prompts: 
- Positive EN (60 batches)
- Negative EN (12 batches)
- Positive UK (60 batches)
- Negative UK (12 batches)
  
Positive samples included sentances containig mountains and Negative samples includes sentances with rivers, cities or metaphorical uses of nature terms.

Each batch (API call) was expected to give 20 sentances.

In total there were generated 144 batches of sentances, giving a total of 2844 samples (after removing duplicates)

The (positive:negative) ratio is important in context of NER. The negative part has to dominate the dataset. Therefore, a ratio of 1:5 was chosen.
##### Alignment logic
The raw text didn't come with token-level labels. Here is the example of generated sentance from LLM:
```
[{'text': 'Mount Everest, also known as Chomolungma, is the highest peak in the world.', 'entities': ['Mount Everest', 'Chomolungma']}]
```
I wrote a custom aligner script that takes the list of entities provided by the LLM and maps them onto the tokenized sentances using a sliding window approach. This aligns the correct BIO tags.

#### 2. Model Architecture
I selected 'bert-base-multilingual-cased' as the base model. This allowed the support for both English and Ukrainian languages. It also handles capitalization.

#### 3. Inference Pipeline Design
I wrote a custom MountainNER class to handle the complexity of sub-word tokenization and giving an intuaitive interface to work with the model.

**Link to trained model weights**: https://huggingface.co/nikolai-domashenko/mnt-ner-model

### Project Structure
- *data/final/:* Contains the processed JSONL datasets (train, test, and validation splits).
- *notebooks/data_preparation.ipynb:* The notebook used for generating data.
- *notebooks/demo.ipynb:* An interactive walkthrough of the model. It includes visualization of predictions and an analysis of where the model fails.
- *src/train.py:* The training script for fine-tuning the model.
- *src/inference.py:* The main inference class containing the model loading, prediction, and evaluation logic.

### Setup and Installation
You will need Python 3.9 or higher. While a powerful GPU is highly recommended for model training, the inference script runs efficiently on a standard CPU.

#### Installation Steps
1. Clone the repository to your local machine:

```
Bash

git clone https://github.com/theairo/ds-internship-test
cd ds-internship-test/task1
```

2. Install the required dependencies: Use the provided requirements.txt file to install all necessary Python packages, including transformers, datasets, pandas, and the google-genai SDK.

```
Bash

pip install -r requirements.txt
```
### Usage
You can interact with the model directly from the terminal.

To predict entities in a single sentence:
```
Bash

python src/inference.py --predict "My dream is to climb Mount Everest."
```
To run the full evaluation suite on the test set:

```
Bash

python src/inference.py --test
```
### Performance
<img width="351" height="162" alt="image" src="https://github.com/user-attachments/assets/46f2ce73-d363-4025-a11f-7238d46eaaf8" />

### Conclusion & Analysis

#### Performance Insights
- The model is trustworthy (precision 0.95 in English) - rarely produces false positives.
- The model performs better on English (F1 0.93) than on Ukrainian (F1 0.88). This is because BERT model was pre-trained on a significantly larger corpus of English text.
- In Ukrainian, Precision and Recall are identical (0.88). This indicates the model is just as likely to miss a mountain as it is to hallucinate one.

#### Error Insights
- The model incorrectly identified Карпати as a mountain when it referred to the football club. (overfitted to the token, lack of negative samples)
- The model missed an instance of "Mauna Kea" in wiki stress test. This proves that the model is imperfect.

#### Conclusion
##### 1. Performance Summary
- The model achieved an overall F1-score of 0.91 across 569 test entities. 
 English Performance (F1: 0.93): The model is highly reliable for English text.
- Ukrainian Performance (F1: 0.88): While effective, the model exhibits a slight performance drop in Ukrainian.

##### 2. Key Strengths
- High Precision (0.95 Overall): The model rarely produces false positives in standard contexts.
- Long-Context Handling: The model successfully tracks multiple entities within dense paragraphs without losing coherence.
