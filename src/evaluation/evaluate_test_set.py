import os
import torch
import pandas as pd
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import argparse

from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.utils.dataset import create_dataloaders
from src.evaluation.classification_metrics import evaluate_model
from src.evaluation.iou import compute_iou
from src.evaluation.pointing_game import compute_pointing_game
from src.xai.gradcam import generate_gradcam
from src.xai.lime_explainer import generate_lime
from src.xai.shap_explainer import generate_shap

def run_evaluation(config_path: str):
    """
    Master Evaluation Script for the Thesis.
    Computes global classification metrics, and if ground-truth masks are provided,
    computes Mean IoU and Pointing Game Accuracy for Grad-CAM, LIME, and SHAP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Dataloaders
    _, _, test_loader = create_dataloaders(config)

    # Load Model
    model_config = config.get('model', {})
    arch = model_config.get('architecture', 'efficientnet_b3')
    num_classes = model_config.get('num_classes', 7)
    
    if arch == 'resnet50':
        model = build_resnet50(num_classes=num_classes, pretrained=False)
        target_layer = model.layer4[-1]
    else:
        model = build_efficientnet_b3(num_classes=num_classes, pretrained=False)
        target_layer = model.features[-1]

    # Load weights
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'outputs/models/')
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Warning: best_model.pth not found. Evaluating with untrained weights!")

    model = model.to(device)
    model.eval()

    # Load SHAP Background
    bg_path = os.path.join(checkpoint_dir, "shap_background.pt")
    if os.path.exists(bg_path):
        shap_bg = torch.load(bg_path, map_location=device)[:100]
    else:
        print("Warning: shap_background.pt not found. Using random noise for SHAP.")
        shap_bg = torch.randn(5, 3, 224, 224).to(device)

    # 1. Classification Metrics
    print("\n--- Running Global Classification Metrics ---")
    class_metrics = evaluate_model(model, test_loader, num_classes)

    # 2. XAI Metrics
    print("\n--- Running XAI Metrics (IoU & Pointing Game) ---")
    # Check if the test DataFrame has a 'mask_path' column
    df_test = test_loader.dataset.data_frame
    if 'mask_path' not in df_test.columns:
        print("No 'mask_path' column found in the test CSV. Skipping XAI spatial evaluation.")
        print("To evaluate XAI, ensure your test CSV links the ISIC 2018 image to its expert mask.")
        return

    xai_results = {
        'gradcam_iou': [], 'gradcam_pg': [],
        'lime_iou': [], 'lime_pg': [],
        'shap_iou': [], 'shap_pg': []
    }

    # Parameters
    lime_samples = config.get('xai', {}).get('lime_num_samples', 1000)
    threshold = config.get('xai', {}).get('attribution_threshold', 0.2)
    
    # Process each image individually
    for idx in tqdm(range(len(test_loader.dataset)), desc="Evaluating XAI"):
        image_tensor, label, img_path = test_loader.dataset[idx]
        mask_path = df_test.iloc[idx]['mask_path']
        
        # Ensure mask exists
        if pd.isna(mask_path) or not os.path.exists(mask_path):
            continue
            
        # Load mask (assume binary or 0-255 grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Resize mask to match image tensor size (224x224 usually)
        _, h, w = image_tensor.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask > 127).astype(np.uint8)

        # Prepare model inputs
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Get target class (we use the predicted class for explainability, or ground truth depending on paradigm)
        # Standard approach: Explain the model's prediction.
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = outputs.argmax(dim=1).item()

        # Generate XAI Maps
        # Grad-CAM
        gradcam_map = generate_gradcam(model, input_tensor, pred_class, target_layer)
        
        # LIME (Needs (H, W, C) float array)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        lime_map = generate_lime(model, image_np, pred_class, num_samples=lime_samples)
        
        # SHAP
        try:
            shap_map = generate_shap(model, input_tensor, shap_bg)
        except Exception:
            shap_map = np.zeros_like(gradcam_map)

        # Compute Metrics
        xai_results['gradcam_iou'].append(compute_iou(gradcam_map, mask_binary, threshold))
        xai_results['gradcam_pg'].append(compute_pointing_game(gradcam_map, mask_binary))
        
        xai_results['lime_iou'].append(compute_iou(lime_map, mask_binary, threshold))
        xai_results['lime_pg'].append(compute_pointing_game(lime_map, mask_binary))
        
        xai_results['shap_iou'].append(compute_iou(shap_map, mask_binary, threshold))
        xai_results['shap_pg'].append(compute_pointing_game(shap_map, mask_binary))

    # Aggregate XAI Results
    if len(xai_results['gradcam_iou']) > 0:
        aggregated_xai = {
            'Mean_GradCAM_IoU': np.mean(xai_results['gradcam_iou']),
            'GradCAM_Pointing_Game_Acc': np.mean(xai_results['gradcam_pg']),
            'Mean_LIME_IoU': np.mean(xai_results['lime_iou']),
            'LIME_Pointing_Game_Acc': np.mean(xai_results['lime_pg']),
            'Mean_SHAP_IoU': np.mean(xai_results['shap_iou']),
            'SHAP_Pointing_Game_Acc': np.mean(xai_results['shap_pg']),
        }
        
        print("\n--- Final XAI Metrics ---")
        for k, v in aggregated_xai.items():
            print(f"{k}: {v:.4f}")
            
        # Merge dicts
        final_results = {**class_metrics, **aggregated_xai}
    else:
        print("No valid mask_paths processed. Only saving classification metrics.")
        final_results = class_metrics

    # Save Master CSV
    os.makedirs('outputs/results', exist_ok=True)
    df = pd.DataFrame([final_results])
    save_path = 'outputs/results/thesis_final_results.csv'
    df.to_csv(save_path, index=False)
    print(f"\nSaved comprehensive thesis results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Evaluation Script")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    run_evaluation(args.config)
