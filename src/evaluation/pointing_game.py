import numpy as np

def compute_pointing_game(attribution_map: np.ndarray, seg_mask: np.ndarray) -> int:
    """
    Compute the Pointing Game metric for an XAI attribution map.
    Checks if the argmax (peak attribution) falls within the ground truth segmentation mask.

    Args:
        attribution_map (np.ndarray): The 2D continuous attribution map.
        seg_mask (np.ndarray): The 2D ground truth binary segmentation mask (0s and 1s).

    Returns:
        int: 1 if the maximum attribution pixel falls inside the mask, else 0.
    """
    assert attribution_map.shape == seg_mask.shape, "Attribution map and segmentation mask must have the same shape."
    
    # Find the index of the maximum value in the flattened array
    max_idx = np.argmax(attribution_map)
    
    # Convert flattened index to 2D coordinates
    max_y, max_x = np.unravel_index(max_idx, attribution_map.shape)
    
    # Check if the mask is positive at that coordinate
    hit = int(seg_mask[max_y, max_x] > 0)
    
    return hit

if __name__ == "__main__":
    # Test Pointing Game
    dummy_map = np.zeros((224, 224))
    dummy_map[120, 120] = 1.0  # Peak inside the mask
    dummy_mask = np.zeros((224, 224))
    dummy_mask[100:150, 100:150] = 1
    
    hit = compute_pointing_game(dummy_map, dummy_mask)
    print(f"Test Pointing Game Hit: {hit}")
