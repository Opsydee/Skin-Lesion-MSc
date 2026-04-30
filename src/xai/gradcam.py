import torch
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, target_class: int, target_layer: nn.Module) -> np.ndarray:
    """
    Generate a Grad-CAM attribution map.

    Args:
        model (nn.Module): The trained PyTorch model.
        image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, C, H, W).
        target_class (int): The target class index for explanation.
        target_layer (nn.Module): The specific layer in the model to compute Grad-CAM on (e.g. model.layer4[-1]).

    Returns:
        np.ndarray: Upsampled attribution map of same spatial size as input, normalized to 0-1.
    """
    model.eval()

    # Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Define the target metric (the specific class we are trying to explain)
    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    # grayscale_cam is shape (batch_size, height, width)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    # We only have a batch size of 1
    attribution_map = grayscale_cam[0, :]
    
    return attribution_map

if __name__ == "__main__":
    from src.models.resnet50 import build_resnet50
    
    # Test with dummy data
    dummy_model = build_resnet50(num_classes=7)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Target the last layer of layer4
    target_layer = dummy_model.layer4[-1]
    
    cam_map = generate_gradcam(dummy_model, dummy_input, target_class=0, target_layer=target_layer)
    print(f"Grad-CAM map generated with shape: {cam_map.shape}, min: {cam_map.min()}, max: {cam_map.max()}")
