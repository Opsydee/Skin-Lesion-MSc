import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Union

def compute_class_weights(labels: Union[List[int], np.ndarray]) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets using scikit-learn.

    Args:
        labels (list or np.ndarray): A list or array of integer class labels for the training set.

    Returns:
        torch.Tensor: A 1D tensor containing the computed class weights, suitable for CrossEntropyLoss.
    """
    labels_np = np.array(labels)
    unique_classes = np.unique(labels_np)
    
    # Compute balanced class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels_np
    )
    
    # Convert to float32 tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute class weights")
    parser.add_argument("--labels", nargs='+', type=int, required=True, help="List of class labels")
    args = parser.parse_args()
    
    weights = compute_class_weights(args.labels)
    print("Computed weights:", weights)
