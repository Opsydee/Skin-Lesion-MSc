import torch
import torch.nn as nn
import numpy as np
import shap

def generate_shap(model: nn.Module, image_tensor: torch.Tensor, background_tensor: torch.Tensor) -> np.ndarray:
    """
    Generate a DeepSHAP attribution map.

    Args:
        model (nn.Module): The trained PyTorch model.
        image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, C, H, W) to explain.
        background_tensor (torch.Tensor): Representative background samples of shape (N, C, H, W) for reference.

    Returns:
        np.ndarray: Absolute SHAP values summed across color channels, normalized 0-1.
    """
    model.eval()
    
    # Create DeepExplainer
    # DeepExplainer takes the model and the background tensor
    explainer = shap.DeepExplainer(model, background_tensor)
    
    # Compute SHAP values for the single input image
    # Note: shap_values is typically a list of arrays (one for each class), 
    # or a single array depending on the SHAP version and model.
    # For a PyTorch multi-class classifier, it's usually a tensor/array or list.
    shap_values = explainer.shap_values(image_tensor)
    
    # Assuming shap_values is a list of arrays of shape (1, C, H, W) for each class
    # To get a single attribution map, we usually take the absolute values,
    # sum across the color channels (C), and normalize.
    
    # If shap_values is a list, we sum across all classes or take the max?
    # Usually we want the SHAP values for the predicted/target class.
    # But since target_class is not passed in the signature required by PRD:
    # "Function: generate_shap(model, image_tensor, background_tensor) -> np.ndarray"
    # We will compute the magnitude across all classes or simply use absolute values.
    # Let's sum absolute SHAP values across all classes and channels.
    
    if isinstance(shap_values, list):
        # List of numpy arrays, one per class.
        shap_values_stacked = np.stack(shap_values, axis=0) # (num_classes, 1, C, H, W)
        shap_abs = np.abs(shap_values_stacked)
        # Sum across classes and channels
        # Shape becomes (1, H, W) -> (H, W)
        attribution_map = np.sum(shap_abs, axis=(0, 2))[0]
    else:
        # Array of shape (1, num_classes, C, H, W) or (1, C, H, W)
        shap_abs = np.abs(shap_values)
        if len(shap_abs.shape) == 5:
            attribution_map = np.sum(shap_abs, axis=(1, 2))[0] # (H, W)
        elif len(shap_abs.shape) == 4:
            attribution_map = np.sum(shap_abs, axis=1)[0] # (H, W)
        else:
            raise ValueError(f"Unexpected shape of shap_values: {shap_abs.shape}")

    # Normalize between 0 and 1
    v_min, v_max = attribution_map.min(), attribution_map.max()
    if v_max > v_min:
        attribution_map = (attribution_map - v_min) / (v_max - v_min)
    else:
        attribution_map = np.zeros_like(attribution_map)
        
    return attribution_map

if __name__ == "__main__":
    from src.models.resnet50 import build_resnet50
    dummy_model = build_resnet50(num_classes=7)
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_background = torch.randn(10, 3, 224, 224) # 10 background samples
    
    # DeepExplainer requires warnings to be suppressed sometimes, but this is a test.
    shap_map = generate_shap(dummy_model, dummy_input, dummy_background)
    print(f"SHAP map generated with shape: {shap_map.shape}, min: {shap_map.min()}, max: {shap_map.max()}")
