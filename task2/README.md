# Cross-Seasonal Satellite Image Matching (Sentinel-2)
This repository contains a computer vision system of matching features across different seasonal conditions (Winter vs. Summer) in Sentinel-2 satellite imagery.

The project demonstrates a fine-tuned SuperGlue model (with frozen SuperPoint weights) compared to benchmarks of State-of-the-Art (SOTA) models in image matching.

## Project Overview
The goal of this task was to fine-tune matching model to a domain specific task (satellite imaging) and compare to benchmarks.

The fine-tuning involved optimizing the SuperGlue Graph Neural Network component while freezing the weights of the SuperPoint feature extractor. This approach allowed for a direct quantitative comparison of the fine-tuned model's ability to handle cross-seasonal domain shifts against State-of-the-Art (SOTA) methods like LoFTR.

## Solution Approach and Design Decisions
### 1. Dataset Preparation

#### Data acquisition

The satellite images were retrived from Copernious Dataspace
https://browser.dataspace.copernicus.eu

The dataset was derived from 4 specific Sentinel-2 tiles covering Ukraine:

- T36UUA (Kyivska oblast)

- T35UNQ (Khmelnytskska oblast - Test Set)

- T36UUU (Kirovohradska oblast)

- T36UVC (Chernihivska oblast)
  
Samples included:
- 1 Winter tile
- 2 Summer tiles (from different years)

Training samples were split into 2 pairs:
- Winter-summer pair
- Summer-summer pair

You can download (split to 512x512) dataset from:
- Google Drive: https://drive.google.com/file/d/1DYFGqjQOxrm4H8tT21Rv5NjJFCxwKIzG/view?usp=sharing

- Kaggle: https://www.kaggle.com/datasets/nikolaidomashenko/superglue-fine-tuning-satellite-image-matching

<img width="1990" height="744" alt="image" src="https://github.com/user-attachments/assets/ecc68540-5157-45b1-9dc9-bcf458a44e69" />


#### Preprocessing
Nomalization strategy:
- Winter Threshold: 8000 (to account for high surface reflectance of snow)
- Summer Threshold: 3000 (standard reflectance)

The massive scale and high dynamic range of raw Sentinel-2 data (100km x 100km tiles, 16-bit depth) was a problem. To solve this, I built a custom preprocessing pipeline using rasterio which splits the large satellite image into 512x512px parts. One large tile produced 441 subtiles.

#### Data stats

In total:
- 2184 pairs were used for training
- 462 pairs were used for validation
- 441 pairs were used for testset

### 2. Model Architecture & Training

#### Model choice
I selected SuperGlue (a Graph Neural Network) operating on top of SuperPoint descriptors as the primary architecture. This allows for sparse, efficient feature matching suitable for georeferencing tasks. I compared this against LoFTR (a dense, detector-free transformer) to benchmark performance.

#### Training approach
##### Transfer Learning with Frozen Weights
We adopted a transfer learning approach by using the pre-trained SuperPoint network as a feature extractor. The weights of SuperPoint model were completely frozen during training. This reduces the computational graph.
   
##### Handling Variable Keypoint Counts

A signifcinant challenge was the inherently variable nature of keypoint detection. Unlike stardard CNN classification where input tensors have fixed dimensions, SuperPoint detects a different number of keypoints N for every image. One image might have 150 keypoints, while another has 300.

Standard PyTorch DataLoaders can't simply stack these variable tensors into a single batch tensor for parallel processing.

To fix this we implemented a custom training loop with Gradient Accumulation:
- The DataLoader fetches a batch of images.
- We iterate through the batch, feeding samples individually into SuperGlue
- Gradients are accumulated for the whole batch before the optimizer performs a step.

*Trade-off*: While this ensures correct processing of sparse graph data without complex padding masks, it introduces a bottleneck. The inability to fully parallelize the SuperGlue forward pass across a large batch makes training significantly slower compared to standard dense tensor operations.

##### Self-Supervised Training Strategy
To overcome the lack of manually labeled keypoints between seasons, we employed a self-supervised training strategy. We applied synthetic homographic warps to the Summer images during training. This allowed us to mathematically calculate the exact "ground truth" geometric relationship between the input pairs, enabling the model to learn precise feature matching from raw unlabeled data without human annotation.

##### Domain-Specific Augmentation
Satellite tiles are always fixed to a North-Up orientation. Therefore, we excluded rotational perturbations from our data augmentation, limiting distortions to perspective tilts (simulating off-nadir sensor angles) and translations.

##### Training process

Since the training requires a high load on compute power, it was orchestrated on Kaggle website (using free-tier GPU T4 x2 with 16GB VRAM). Here is the link to the Kaggle notebook:
https://www.kaggle.com/code/nikolaidomashenko/satellite-image-matching-fine-tuning-superglue

**Link to final model weights:**
https://huggingface.co/nikolai-domashenko/superglue-sat

### 3. Inference Pipeline Design
I implemented a modular inference pipeline designed for benchmarking. Unlike standard scripts that simply output raw feature pairs, this pipeline integrates a geometric verification step using Homography estimation with RANSAC to calculate Precision (percentage of geometrically consistent inliers).

The pipeline exposes a dynamic threshold parameter. This allows users to perform sensitivity analysis, sweeping through different reprojection error tolerances (e.g., 3px, 5px, 10px) to quantify the trade-off between match quantity (Recall) and localization accuracy (Precision) for each model.

## Project Structure
- *dataset_final/:* Contains the processed PNG subtiles organized by Tile ID and Season (note: the dataset in repo is cutoff (only includes first 5 subtiles from each tile) to ensure demo testing.
- *notebooks/dataset_creation.ipynb:* The notebook used for processing raw .jp2 satellite bands into paired dataset images.
- *notebooks/demo.ipynb:* An interactive walkthrough of the model. It includes Sensitivity Analysis curves and visualization of prediction grids.
- *src/train.py:* The training script for fine-tuning the SuperGlue model.
- *src/inference.py:* The main inference class containing model loading, benchmarking logic, and visualization tools.
- *assets/:* Contains satellite images for demo.
- *src/external/:* The SuperGlue essentials for training the model.
- *output/:* Saved visualized matches.

## Setup and Installation
You will need Python 3.9 or higher. While a powerful GPU is highly recommended for model training, the inference script runs efficiently on a standard CPU.

### Installation Steps
1. Clone the repository to your local machine:

```
Bash

git clone https://github.com/theairo/ds-internship-test
cd ds-internship-test/task2
```

2. Install the required dependencies: Use the provided requirements.txt file to install all necessary Python packages, including transformers, datasets, pandas, and the google-genai SDK.

```
Bash

pip install -r requirements.txt
```
## Usage
You can interact with the model directly from the terminal.

To visualize predictions for a specific subtile index (saves plot to output/):

To predict entities in a single sentence:
```
Bash

python src/inference.py --predict 5
```
To run the full evaluation suite on the test set:

```
Bash

python src/inference.py --test
```

## Demo
<img width="1089" height="1814" alt="image" src="https://github.com/user-attachments/assets/4de51bb7-aef7-4917-a2f7-9d9ff092ec88" />


## Performance
<img width="1463" height="627" alt="image" src="https://github.com/user-attachments/assets/c4b31c81-5696-4dc3-86a8-3a73e79baabb" />

## Final Analysis:
We benchmarked three models on Sentinel-2 satellite imagery, evaluating their robustness against seasonal appearance changes (Snow vs Summer). The sensitivity analysis reveals three key findings:

1. The Dominance of Dense Matching (LoFTR)
LoFTR establishes the upper bound for performance, achieving:

- Highest Recall: ~1,300 matches per image (4x more than sparse methods).

- Highest Stability: >95% precision even at 3px threshold.

- Why: Detector-free architecture allows it to find correspondences in low-texture areas (snow fields) where detector-based methods (SuperPoint) fail to find interest points. It also performs sub-pixel refinement, making it stable against localization jitter.

2. The Fine-Tuning Trade-off
Comparing the Pretrained vs Fine-tuned SuperGlue reveals a behavior change:

- Recall Gain: The fine-tuning successfully taught the model to bridge the gap between seasons. It finds ~25% more matches (380 vs. 300) than the pretrained weights.

- Precision Drop at Low Tolerance (trade-off): At low thresholds (<5px), the fine-tuned model’s precision drops below the pretrained model.

- Possible interpretation: The pretrained model only matches the "easy," distinct features, which are spatially accurate. The fine-tuned model is aggressively matches seasonal features.

3. Conclusion
- For satellite georeferencing where a 10px error (100m) is often within the acceptable margin for coarse alignment, the Fine-tuned SuperGlue is superior to the pretrained version because it provides a denser graph of connections.

- However, the drop in precision at 3px indicates that SuperPoint's keypoint detection (which we froze during the training) is the bottleneck for pixel-perfect accuracy on snow, not the SuperGlue matching logic itself.
