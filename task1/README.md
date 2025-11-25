# Phase 1: Proof of Concept

## Model & Dataset
- Dataset: 240 samples (1:1 Positive/Negative ratio)
- Model: bert-base-cased (Pre-trained)

## Training Details
- Training Time: ~2 minutes (5 epochs)
- Dataset acquisition:
    - Total time: ~4 minutes
    - API calls: 13 (1 failed: "Model is overloaded")

## Evaluation
- Precision: 0.8333
- Recall:    0.9375
- F1 Score:  0.8824
- Loss:      0.0700

## Inference Test
- Sentence: 'Mount Rainier is visible from Seattle on clear days.'
- Result:

[
  {"entity_group": "MNT", "score": 0.9931, "word": "Mount", "start": 0, "end": 5},
  {"entity_group": "MNT", "score": 0.9934, "word": "Rainier", "start": 6, "end": 13}
]

## Known Limitations
- Multi-word entity splitting: Currently, "Mount Rainier" is recognized as two separate entities.
- Potential cause: Small dataset (120 samples) and insufficient I-MNT examples.
- Planned Fix: Expand dataset to ~1,000 samples in Phase 2.


PHASE 2

{'loss': 0.0011, 'grad_norm': 0.10494685173034668, 'learning_rate': 2.3419203747072602e-08, 'epoch': 5.0}                                                                
{'eval_loss': 0.02149352617561817, 'eval_runtime': 2.862, 'eval_samples_per_second': 198.812, 'eval_steps_per_second': 49.965, 'epoch': 5.0}                             
{'train_runtime': 353.8103, 'train_samples_per_second': 24.109, 'train_steps_per_second': 6.034, 'train_loss': 0.044171580116251434, 'epoch': 5.0}                       
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2135/2135 [05:53<00:00,  6.03it/s]