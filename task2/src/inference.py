import os
import sys
import cv2
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import kornia.feature as KF
from huggingface_hub import hf_hub_download
import argparse
import pandas as pd

script_dir = Path(__file__).parent.resolve()
project_dir = script_dir.parent
data_dir = project_dir / "dataset_final"
output_dir = project_dir / "output"

# Add the project root to sys.path
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))
    print(f"Added to path: {project_dir}")

class SuperGlueEvaluator:
    def __init__(self, weights_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from src.external.SuperGluePretrainedNetwork.models.matching import Matching
        except ImportError:
            raise ImportError("Could not import Matching from SuperGluePretrainedNetwork. Check paths.")

        self.config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024,
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        
        self.model = Matching(self.config).eval().to(self.device)

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.superglue.load_state_dict(state_dict)
            print(f"Loaded Custom Weights: {os.path.basename(weights_path)}")
        else:
            print("Loaded Default Weights")

    def match(self, img0_tensor, img1_tensor):
        with torch.no_grad():
            pred = self.model({'image0': img0_tensor, 'image1': img1_tensor})
        
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        
        # Filter valid matches
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        return mkpts0, mkpts1

class LoFTRKornia:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KF.LoFTR(pretrained='outdoor').to(self.device).eval()

    def match(self, img0_tensor, img1_tensor):
        with torch.no_grad():
            pred = self.model({'image0': img0_tensor, 'image1': img1_tensor})
        
        kpts0 = pred["keypoints0"].cpu().numpy()
        kpts1 = pred["keypoints1"].cpu().numpy()
        confidence = pred["confidence"].cpu().numpy()

        mask = confidence > 0.5
        return kpts0[mask], kpts1[mask]

def compute_metrics_honest(mkpts0, mkpts1, threshold = 3.0):
    """
    Calculates metrics based on RANSAC consensus.
    Does NOT return 'Recall' because we don't have Ground Truth.
    """
    if len(mkpts0) < 4:
        return None

    H, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, threshold)
    
    if H is None:
        return None

    inliers = inlier_mask.ravel().astype(bool)
    num_inliers = int(inliers.sum())
    num_matches = len(mkpts0)
    
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0

    return {
        "num_matches": num_matches,
        "num_inliers": num_inliers,  
        "inlier_ratio": inlier_ratio, 
        "H": H,
        "inliers_mask": inliers
    }


def evaluate_model(model, dataset_path, tile_code, device, threshold, max_pairs=30000):
    winter_dir = os.path.join(dataset_path, tile_code, "winter")
    summer_dir = os.path.join(dataset_path, tile_code, "summer")
    
    files = sorted(glob.glob(os.path.join(winter_dir, "*.png")))[:max_pairs]
    
    metrics_history = {"num_matches": [], "num_inliers": [], "inlier_ratio": []}
    
    print(f"\n- Evaluating {type(model).__name__} on {tile_code} -")
    
    for path_winter in tqdm(files):
        filename = os.path.basename(path_winter)
        path_summer = os.path.join(summer_dir, filename)
        
        if not os.path.exists(path_summer): continue
        
        # Load Images
        img0_gray = cv2.imread(path_winter, cv2.IMREAD_GRAYSCALE)
        img1_gray = cv2.imread(path_summer, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        t0 = torch.from_numpy(img0_gray/255.0).float()[None, None].to(device)
        t1 = torch.from_numpy(img1_gray/255.0).float()[None, None].to(device)
        
        # Run Matcher
        mkpts0, mkpts1 = model.match(t0, t1)
        
        # Compute Metrics
        res = compute_metrics_honest(mkpts0, mkpts1, threshold=threshold)
        
        if res:
            metrics_history["num_matches"].append(res["num_matches"])
            metrics_history["num_inliers"].append(res["num_inliers"])
            metrics_history["inlier_ratio"].append(res["inlier_ratio"])

    # Aggregate
    if not metrics_history["num_matches"]:
        print("No matches found in dataset.")
        return

    avg_matches = np.mean(metrics_history["num_matches"])
    avg_inliers = np.mean(metrics_history["num_inliers"])
    avg_ratio = np.mean(metrics_history["inlier_ratio"])
    
    print(f"Results:")
    print(f"  Avg Matches:     {avg_matches:.1f}")
    print(f"  Avg Inliers:     {avg_inliers:.1f} ")
    print(f"  Avg Precision:   {avg_ratio:.3f}     (Inlier Ratio)")

    return metrics_history

def draw_matches(img0, img1, kpts0, kpts1, mask, metrics=None, model_name=None, threshold=3.0):
    h, w = img0.shape
    
    # Define a top margin (Header) size
    margin_top = 30  # Height of the black bar
    
    # Create the combined image (Side-by-Side)
    stitched = np.hstack((img0, img1))
    stitched = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGR)
    
    # Create a blank canvas with extra height for the header
    vis_h, vis_w, _ = stitched.shape
    vis = np.zeros((vis_h + margin_top, vis_w, 3), dtype=np.uint8) # Black canvas
    
    # Paste the images onto the canvas (shifted down)
    vis[margin_top:, :] = stitched
    
    # Draw Lines and Dots (Shift Y-coordinates by margin_top)
    for i in range(len(kpts0)):
        # Original coordinates
        x0, y0 = int(kpts0[i][0]), int(kpts0[i][1])
        x1, y1 = int(kpts1[i][0]) + w, int(kpts1[i][1])
        
        # Shifted coordinates
        p0 = (x0, y0 + margin_top)
        p1 = (x1, y1 + margin_top)
        
        if mask[i]:
            color = (0, 255, 0) # Green
            thickness = 1
            # Draw dots on top of the image
            cv2.circle(vis, p0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(vis, p1, 2, color, -1, cv2.LINE_AA)
            # Draw line
            cv2.line(vis, p0, p1, color, thickness, cv2.LINE_AA)
        else:
            color = (0, 0, 255) # Red
            thickness = 1

            cv2.line(vis, p0, p1, color, thickness, cv2.LINE_AA)
            cv2.circle(vis, p0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(vis, p1, 2, color, -1, cv2.LINE_AA)

    # Common Text Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    color_text = (255, 255, 255)
    
    # Draw Metrics Text (In the new Header space)
    if metrics:
        matches_count = metrics['num_matches']
        inliers_count = metrics['num_inliers']
        precision = metrics['inlier_ratio']
        
        text_left = f"Matches: {matches_count} | Inliers: {inliers_count} | Precision: {precision:.2f} | GSD Error: {threshold:.2f}px"
        
        # Center vertically in the margin
        text_y = int(margin_top / 2) + 6
        cv2.putText(vis, text_left, (10, text_y), font, scale, color_text, thickness, cv2.LINE_AA)

    # Draw Model Name (Top-Right of Header)
    if model_name:
        text_right = model_name
        (tw, th), _ = cv2.getTextSize(text_right, font, scale, thickness)
        
        x_pos = vis_w - tw - 10 
        text_y = int(margin_top / 2) + 6
        
        cv2.putText(vis, text_right, (x_pos, text_y), font, scale, color_text, thickness, cv2.LINE_AA)

    return vis

def predict_subtile(subtile_idx, model, dataset_path, tile_code, device, threshold=3.0):
    """
    Runs prediction on a specific subtile index, calculates inliers, 
    and saves a visualization image with metrics written on it.
    """
    # Setup specific paths
    winter_dir = os.path.join(dataset_path, tile_code, "winter")
    summer_dir = os.path.join(dataset_path, tile_code, "summer")
    files = sorted(glob.glob(os.path.join(winter_dir, "*.png")))
    
    if subtile_idx < 0 or subtile_idx >= len(files):
        print(f"Error: Index {subtile_idx} out of bounds.")
        return None

    path_winter = files[subtile_idx]
    filename = os.path.basename(path_winter)
    path_summer = os.path.join(summer_dir, filename)

    if not os.path.exists(path_summer):
        return None

    # Get clean model name
    model_name = getattr(model, 'name', type(model).__name__)
    print(f"\n- Predicting on Subtile #{subtile_idx} using {model_name} -")

    # Load & Preprocess
    img0_gray = cv2.imread(path_winter, cv2.IMREAD_GRAYSCALE)
    img1_gray = cv2.imread(path_summer, cv2.IMREAD_GRAYSCALE)
    t0 = torch.from_numpy(img0_gray/255.0).float()[None, None].to(device)
    t1 = torch.from_numpy(img1_gray/255.0).float()[None, None].to(device)

    # Match
    mkpts0, mkpts1 = model.match(t0, t1)
 
    # Compute Metrics
    res = compute_metrics_honest(mkpts0, mkpts1, threshold=threshold)
    
    # Handle output Stats
    stats = {
        "Model": model_name,
        "Threshold (px)": threshold,
        "Matches": 0,
        "Inliers": 0,
        "Precision": 0.0
    }

    if res is None:
        print("Warning: Not enough matches.")
        mask = np.zeros(len(mkpts0), dtype=bool) 
        # Update stats for partial data (matches exist, geometry failed)
        stats["Matches"] = len(mkpts0)
    else:
        mask = res['inliers_mask']
        # Update stats with real results
        stats["Matches"] = res['num_matches']
        stats["Inliers"] = res['num_inliers']
        stats["Precision"] = res['inlier_ratio']

    # Visualize and Save
    vis = draw_matches(img0_gray, img1_gray, mkpts0, mkpts1, mask, metrics=res, model_name=model_name, threshold=threshold)
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_filename = f"pred_{tile_code}_{subtile_idx}_{safe_name}_{int(threshold)}px.png"
    
    if 'output_dir' not in globals():
        # Fallback if output_dir isn't defined
        save_path = out_filename 
    else:
        save_path = str(output_dir / out_filename)

    cv2.imwrite(save_path, vis)
    print(f"   Visualization saved to: {save_path}")
    
    return stats

def run_inference(args):
    # Setup Device & Load Heavy Weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        repo_id = "nikolai-domashenko/superglue-sat"
        
        # Download weights
        model_weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="superglue-sat.pth" # Ensure this matches your specific filename on HF
        )
        print(f"Model downloaded to: {model_weights_path}")

        # Initialize Models
        # Custom Finetuned SuperGlue
        sg_custom = SuperGlueEvaluator(weights_path=model_weights_path, device=device)
        sg_custom.name = "SuperGlue (Finetuned)"

        # Pretrained SuperGlue
        sg_default = SuperGlueEvaluator(weights_path=None, device=device)
        sg_default.name = "SuperGlue (Pretrained)"
        
        # LoFTR
        loftr = LoFTRKornia(device)
        loftr.name = "LoFTR (Kornia)"

        models = [loftr, sg_default, sg_custom]

    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    results_list = []

    if args.predict is not None:
        print(f"Generating stats table for Subtile {args.predict}...")
        
        for model in models:
            # Capture the return value
            stats = predict_subtile(
                subtile_idx=args.predict, 
                model=model, 
                dataset_path=args.data_path, 
                tile_code=args.test_tile, 
                device=device,
                threshold=args.ransac_thresh
            )
            
            if stats:
                results_list.append(stats)
                
        # Return DataFrame immediately if we did prediction
        if results_list:
            df = pd.DataFrame(results_list)
            return df

    if args.test:
        print(f"\nStarting evaluation on Tile: {args.test_tile} | Path: {args.data_path}")
        
        for model in models:
            try:
                model_name = getattr(model, 'name', str(model))
                print(f"\n- Evaluating {model_name} -")
                
                # Capture the returned history dictionary
                history = evaluate_model(
                    model, 
                    args.data_path, 
                    args.test_tile, 
                    device, 
                    threshold=args.ransac_thresh
                )
                
                # Flatten the data for the DataFrame
                if history:
                    for i in range(len(history["num_matches"])):
                        results_list.append({
                            "Model": model_name,
                            "Matches": history["num_matches"][i],
                            "Inliers": history["num_inliers"][i],
                            "Precision": history["inlier_ratio"][i],
                            "Threshold": args.ransac_thresh
                        })
                
            except Exception as e:
                print(f"{model_name} failed: {e}")

        # Return DataFrame if we gathered data
        if results_list:
            df = pd.DataFrame(results_list)
            return df
            
    # Default return if no action was taken
    return None

if __name__ == "__main__":
    # Keep ONLY the argparse stuff here
    parser = argparse.ArgumentParser(description="Satellite Image Matching Evaluation")
    parser.add_argument("--data_path", type=str, default=data_dir)
    parser.add_argument("--test_tile", type=str, default="T35UNQ")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", type=int)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)

    args = parser.parse_args()
    
    # Call the function
    run_inference(args)