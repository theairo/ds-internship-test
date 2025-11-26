import os
import shutil
import sys
import glob
import random
import json
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# --- Path Configuration ---
# Determine where we are (Script Location)
script_path = Path(__file__).resolve()
script_dir = script_path.parent       # e.g., /MyProject/code/
project_dir = script_dir.parent       # e.g., /MyProject/

# Define Project Paths relative to the script
REPO_DIR_NAME = "SuperGluePretrainedNetwork" 
repo_dir = script_dir / "external" / REPO_DIR_NAME
data_dir = project_dir / "dataset_final"

# Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
MODEL_CONFIG = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 128, # 512
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 5, # 20
        'match_threshold': 0.2,
    }
}

# Imports from Repo
try:
    from external.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
    from external.SuperGluePretrainedNetwork.models.superglue import SuperGlue
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure your repo folder contains a 'models' directory with superglue.py inside.")
    sys.exit(1)


# ==========================================
# AUGMENTATION UTILS
# ==========================================

def sample_homography(shape, perturb_factor=0.1):
    """
    Generates a random homography matrix for augmentation.
    FOCUS: Perspective (Camera Tilt) and Translation. NO Rotation (Spin).
    """
    h, w = shape
    
    # Define original corners
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = pts1.copy()

    # Perspective / Affine perturbations
    shift_x = np.random.uniform(-perturb_factor * w, perturb_factor * w, 4)
    shift_y = np.random.uniform(-perturb_factor * h, perturb_factor * h, 4)
    
    pts2[:, 0] += shift_x
    pts2[:, 1] += shift_y

    # Calculate Homography Matrix
    H_mat = cv2.getPerspectiveTransform(pts1, pts2)
    return H_mat

def warp_keypoints(kpts, H):
    """
    Warps keypoints using the homography matrix H.
    """
    # Convert to homogeneous coordinates
    ones = torch.ones(kpts.shape[0], 1, device=kpts.device)
    kpts_homo = torch.cat([kpts, ones], dim=1)
    
    # Apply transformation: p' = H * p
    kpts_warped = (H @ kpts_homo.T).T
    
    # Normalize back to Cartesian (x/z, y/z)
    kpts_warped = kpts_warped[:, :2] / kpts_warped[:, 2:]
    return kpts_warped


# ==========================================
# DATASET CLASS
# ==========================================

class SeasonalDataset(Dataset):
    def __init__(self, root_dir, allowed_tiles=None, split_mode='train', split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.samples = [] 
        
        # Define Pair Rules
        pair_rules = [
            ("winter", "summer"),
            ("summer", "summer_prev")
        ]
        
        # Auto-detect tiles if None
        if allowed_tiles is None:
            if os.path.exists(root_dir):
                allowed_tiles = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            else:
                allowed_tiles = []
                print(f"Warning: Dataset path {root_dir} does not exist.")
        
        print(f"--- Loading data for: {split_mode} (Ratio: {split_ratio}) ---\n")
        
        # Collect ALL valid pairs first
        all_valid_pairs = []

        for tile_code in allowed_tiles:
            tile_path = os.path.join(root_dir, tile_code)
            
            for folder_a, folder_b in pair_rules:
                dir_a = os.path.join(tile_path, folder_a)
                dir_b = os.path.join(tile_path, folder_b)
                
                if not os.path.exists(dir_a) or not os.path.exists(dir_b): continue
                
                files_a = glob.glob(os.path.join(dir_a, "*.png"))
                
                for path_a in files_a:
                    filename = os.path.basename(path_a) 
                    path_b = os.path.join(dir_b, filename)
                    
                    # Only add if both exist
                    if os.path.exists(path_b):
                        all_valid_pairs.append((path_a, path_b))

        # Shuffle Deterministically
        all_valid_pairs.sort() 
        random.seed(seed)
        random.shuffle(all_valid_pairs)

        # Apply the Split
        num_samples = len(all_valid_pairs)
        split_idx = int(num_samples * split_ratio)

        if split_mode == 'train':
            self.samples = all_valid_pairs[:split_idx]
        elif split_mode == 'val':
            self.samples = all_valid_pairs[split_idx:]
        else: # 'full'
            self.samples = all_valid_pairs

        print(f"   -> Total pairs found: {num_samples}")
        print(f"   -> Selected for {split_mode}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_a, path_b = self.samples[idx]
        
        img0 = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        
        if img0 is None or img1 is None:
            # Return dummy data if read fails
            return torch.zeros(1, 480, 480), torch.zeros(1, 480, 480), torch.eye(3)

        # Augmentation
        height, width = img1.shape
        H_mat = sample_homography((height, width)) 
        img1_warped = cv2.warpPerspective(img1, H_mat, (width, height))
        
        t0 = torch.from_numpy(img0.astype('float32') / 255.0)
        t1 = torch.from_numpy(img1_warped.astype('float32') / 255.0)
        tH = torch.from_numpy(H_mat.astype('float32'))
        
        return t0, t1, tH


# ==========================================
# GEOMETRY & LOSS
# ==========================================

def make_geometric_labels(kpts0, kpts1, H, threshold=4.0):
    """
    Computes matches by projecting kpts0 into image1 using H 
    and checking distance to kpts1.
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return torch.tensor([], dtype=torch.long, device=kpts0.device)

    # Project kpts0 using the Ground Truth Homography
    kpts0_projected = warp_keypoints(kpts0, H)

    # Compute Distance (Projected Kpts0 vs Actual Kpts1)
    dist = torch.cdist(kpts0_projected, kpts1, p=2)
    
    # Find Matches
    min_dist, nn_idx = torch.min(dist, dim=1)
    mask = min_dist < threshold
    
    matches0 = torch.full((kpts0.shape[0],), -1, dtype=torch.long, device=kpts0.device)
    matches0[mask] = nn_idx[mask]
    
    return matches0

def compute_superglue_loss(pred_scores, gt_matches0, device):
    scores = pred_scores[0] # Remove batch dim
    
    N = gt_matches0.shape[0]
    targets = gt_matches0.clone()
    dustbin_index = scores.shape[1] - 1 
    
    targets[targets == -1] = dustbin_index
    
    # Only compute loss for the N detected keypoints
    valid_scores = scores[:N, :] 
    
    loss = nn.functional.nll_loss(valid_scores, targets)
    return loss


# ==========================================
# TRAINING LOOP
# ==========================================

# --Batch Processor (Logic for one batch) --
def process_batch(superglue, superpoint, batch, device, training=True):
    """
    Processes a single batch:
    1. Runs SuperPoint (frozen)
    2. Iterates through items in batch
    3. Runs SuperGlue
    4. Computes Loss
    5. Backpropagates if training
    """
    img0_batch, img1_batch, H_batch = batch
    img0_batch = img0_batch.unsqueeze(1).to(device)
    img1_batch = img1_batch.unsqueeze(1).to(device)
    H_batch = H_batch.to(device)
    
    # Run SuperPoint (Always frozen/no_grad)
    with torch.no_grad():
        out0 = superpoint({'image': img0_batch})
        out1 = superpoint({'image': img1_batch})
    
    batch_loss = 0
    valid_samples = 0
    
    # Inner loop: Handle samples individually (fake batch size = 1 for SuperGlue)
    for b in range(img0_batch.shape[0]):
        kpts0 = out0['keypoints'][b]
        kpts1 = out1['keypoints'][b]
        desc0 = out0['descriptors'][b]
        desc1 = out1['descriptors'][b]
        scores0 = out0['scores'][b]
        scores1 = out1['scores'][b]
        H_current = H_batch[b]

        # Skip if too few keypoints (Satellite images might have sparse features)
        if len(kpts0) < 10 or len(kpts1) < 10:
            continue

        gt_matches0 = make_geometric_labels(kpts0, kpts1, H_current)
        
        # Prepare Input for SuperGlue
        data = {
            'image0': img0_batch[b].unsqueeze(0),
            'image1': img1_batch[b].unsqueeze(0),
            'keypoints0': kpts0.unsqueeze(0),
            'keypoints1': kpts1.unsqueeze(0),
            'descriptors0': desc0.unsqueeze(0),
            'descriptors1': desc1.unsqueeze(0),
            'scores0': scores0.unsqueeze(0),
            'scores1': scores1.unsqueeze(0),
        }
        
        # SuperGlue Forward Pass
        pred = superglue(data)
        
        # Loss Calculation
        loss = compute_superglue_loss(pred['scores'], gt_matches0, device)
        
        # Accumulate Gradient (Gradient Accumulation)
        if training:
            loss.backward()
            
        batch_loss += loss.item()
        valid_samples += 1
    
    return batch_loss, valid_samples

def run_epoch(superglue, superpoint, loader, device, epoch, optimizer=None, training=True):
    if training:
        superglue.train()
        phase_name = "Training"
    else:
        superglue.eval()
        phase_name = "Validation"

    total_loss = 0
    total_batches = 0
    
    with torch.set_grad_enabled(training):
        for i, batch in enumerate(loader):
            if training and optimizer:
                optimizer.zero_grad()
            
            batch_loss_sum, valid_samples = process_batch(
                superglue, superpoint, batch, device, training=training
            )
            
            if training and valid_samples > 0 and optimizer:
                optimizer.step()
            
            if valid_samples > 0:
                avg_batch_loss = batch_loss_sum / valid_samples
                total_loss += avg_batch_loss
                total_batches += 1
                
                if training and i % 50 == 0:
                    print(f"   [{phase_name}] Batch {i} | Loss: {avg_batch_loss:.4f}")
            
    avg_epoch_loss = total_loss / max(1, total_batches)
    return avg_epoch_loss

def main():
    print(f"--- Starting Full Training Run on {DEVICE} ---")
    
    # Model Setup
    superpoint = SuperPoint(MODEL_CONFIG['superpoint']).to(DEVICE)
    superpoint.eval() # Always frozen
    
    superglue = SuperGlue(MODEL_CONFIG['superglue']).to(DEVICE)
    
    optimizer = optim.Adam(superglue.parameters(), lr=LEARNING_RATE)

    # Define specific tiles or leave None for auto-detection
    TILES = ['T36UUA', 'T36UUU', 'T36UVC']   

    print("Initializing Datasets...")
    
    # Create Training Set
    train_ds = SeasonalDataset(data_dir, TILES, split_mode='train', split_ratio=0.8)
    
    # Create Validation Set
    val_ds = SeasonalDataset(data_dir, TILES, split_mode='val', split_ratio=0.8)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        
        train_loss = run_epoch(
            superglue, superpoint, train_loader, DEVICE, epoch, 
            optimizer=optimizer, training=True
        )
        
        val_loss = run_epoch(
            superglue, superpoint, val_loader, DEVICE, epoch, 
            optimizer=None, training=False
        )
        
        print(f"Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        with open('training_log.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        # Save checkpoint
        torch.save(superglue.state_dict(), f"superglue_epoch_{epoch+1}.pth")

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()