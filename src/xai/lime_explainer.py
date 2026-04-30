import torch
import torch.nn as nn
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F

def generate_lime(model: nn.Module, image_np: np.ndarray, target_class: int, num_samples: int = 1000) -> np.ndarray:
    """
    Generate a LIME attribution map.

    Args:
        model (nn.Module): The trained PyTorch model.
        image_np (np.ndarray): Original image as a numpy array in RGB format (H, W, C), values [0, 1] or [0, 255].
                               If values are [0, 255], LIME generally expects them as standard images.
        target_class (int): The target class index for explanation.
        num_samples (int): Number of perturbation samples to generate.

    Returns:
        np.ndarray: Binary superpixel mask as a float attribution map (0 or 1 per superpixel).
    """
    model.eval()
    device = next(model.parameters()).device

    def batch_predict(images_numpy):
        """
        Batch prediction function required by LIME.
        Converts numpy arrays to model inputs and returns class probabilities.
        LIME passes images in format (N, H, W, C) with values [0, 255] or [0, 1].
        """
        model.eval()
        # Convert to tensor and shape (N, C, H, W)
        batch = torch.stack([
            # Assuming LIME inputs need to be transposed to (C, H, W) and normalized if needed
            torch.from_numpy(img.transpose((2, 0, 1))).float()
            for img in images_numpy
        ], dim=0)
        
        batch = batch.to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    # LIME operates on double precision, but we pass standard float numpy array
    explanation = explainer.explain_instance(
        image_np.astype(np.float64), 
        batch_predict, 
        top_labels=None,
        labels=(target_class,),
        hide_color=0, 
        num_samples=num_samples
    )

    # Get image and mask for the top class (target_class)
    # positive_only=True means we only consider superpixels that positively contribute to the target_class
    # num_features=10 (or similar) will get the top superpixels
    # Since we need a full attribution map, let's get the raw local weights per superpixel and threshold them.
    # The requirement is: "Return binary superpixel mask as a float attribution map (0 or 1 per superpixel)"
    
    # We can get the top 20% of superpixels or just all positive superpixels. 
    # Let's get the mask for the top features
    temp, mask = explanation.get_image_and_mask(
        target_class, 
        positive_only=True, 
        num_features=10, 
        hide_rest=False
    )
    
    # mask is an integer array where 1 indicates the presence of a superpixel, 0 otherwise
    attribution_map = mask.astype(np.float32)

    return attribution_map

if __name__ == "__main__":
    from src.models.resnet50 import build_resnet50
    dummy_model = build_resnet50(num_classes=7)
    dummy_image = np.random.rand(224, 224, 3) # random RGB image (0-1)
    
    lime_map = generate_lime(dummy_model, dummy_image, target_class=0, num_samples=10)
    print(f"LIME map generated with shape: {lime_map.shape}, min: {lime_map.min()}, max: {lime_map.max()}")
