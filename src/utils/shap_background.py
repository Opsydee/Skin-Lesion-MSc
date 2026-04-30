import os
import torch
from torch.utils.data import DataLoader

def generate_and_save_background(train_loader: DataLoader, num_samples: int = 100, save_path: str = 'outputs/shap_background.pt'):
    """
    Generate and save a representative background tensor for DeepSHAP.
    
    Args:
        train_loader (DataLoader): The training dataloader.
        num_samples (int): Total number of samples to collect for the background distribution.
        save_path (str): File path to save the tensor.
    """
    print(f"Generating SHAP background tensor with {num_samples} samples...")
    background_samples = []
    collected = 0
    
    for images, _, _ in train_loader:
        batch_size = images.size(0)
        if collected + batch_size <= num_samples:
            background_samples.append(images)
            collected += batch_size
        else:
            remaining = num_samples - collected
            if remaining > 0:
                background_samples.append(images[:remaining])
                collected += remaining
            break
            
    if collected == 0:
        print("Warning: Train loader is empty. Cannot generate background.")
        return
        
    bg_tensor = torch.cat(background_samples, dim=0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(bg_tensor, save_path)
    print(f"Successfully saved SHAP background tensor of shape {bg_tensor.shape} to {save_path}")

if __name__ == "__main__":
    print("This is a utility module to be called by train.py.")
