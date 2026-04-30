import numpy as np

def compute_iou(attribution_map: np.ndarray, seg_mask: np.ndarray, threshold: float = 0.2) -> float:
    """
    Compute Intersection over Union (IoU) between an XAI attribution map and a ground truth segmentation mask.

    Args:
        attribution_map (np.ndarray): The 2D continuous attribution map (values usually 0-1).
        seg_mask (np.ndarray): The 2D ground truth binary segmentation mask (0s and 1s).
        threshold (float): Top fraction of pixels to threshold as positive. Default 0.2 (top 20%).

    Returns:
        float: The IoU score.
    """
    assert attribution_map.shape == seg_mask.shape, "Attribution map and segmentation mask must have the same shape."
    
    # Flatten the map to find the threshold value
    flat_map = attribution_map.flatten()
    
    # Find the value at the specified top percentile
    # If threshold is 0.2, we want the 80th percentile value
    percentile_val = np.percentile(flat_map, 100 * (1.0 - threshold))
    
    # Create binary mask from attribution map
    xai_binary = (attribution_map >= percentile_val).astype(np.uint8)
    
    # Ensure seg_mask is binary (0 or 1)
    seg_binary = (seg_mask > 0).astype(np.uint8)
    
    # Compute Intersection and Union
    intersection = np.logical_and(xai_binary, seg_binary).sum()
    union = np.logical_or(xai_binary, seg_binary).sum()
    
    if union == 0:
        return 0.0
        
    iou = intersection / union
    return float(iou)

if __name__ == "__main__":
    # Test
    dummy_map = np.random.rand(224, 224)
    dummy_mask = np.zeros((224, 224))
    dummy_mask[100:150, 100:150] = 1
    
    iou_score = compute_iou(dummy_map, dummy_mask)
    print(f"Test IoU Score: {iou_score:.4f}")
