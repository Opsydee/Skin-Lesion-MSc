import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from src.preprocessing.hair_removal import remove_hair
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms

class SkinLesionDataset(Dataset):
    """
    Custom Dataset for Skin Lesion Images.
    Loads images from a CSV file, optionally applies hair removal, and applies transforms.
    """
    def __init__(self, csv_path: str, transform=None, apply_hair_removal: bool = False):
        """
        Args:
            csv_path (str): Path to the CSV file containing 'image_path' and 'label'.
            transform (callable, optional): Optional transform to be applied on a sample.
            apply_hair_removal (bool): Whether to apply morphological hair removal.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform
        self.apply_hair_removal = apply_hair_removal

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['image_path']
        label = int(self.data_frame.iloc[idx]['label'])

        # Load image with OpenCV (BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")

        # Apply hair removal if enabled
        if self.apply_hair_removal:
            image = remove_hair(image)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (which expect a PIL Image or numpy array depending on setup)
        # Our transforms expect numpy array or PIL image. ToPILImage() is the first step in transforms.
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Fallback if no transform is provided
            # Convert to tensor and scale to [0, 1]
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        return image_tensor, label, img_path

def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Train, Validation, and Test dataloaders based on the provided configuration.

    Args:
        config (dict): Configuration dictionary usually loaded from config.yaml.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """
    data_config = config.get('data', {})
    
    image_size = data_config.get('image_size', 224)
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    apply_hair_removal = data_config.get('apply_hair_removal', False)
    
    train_csv = data_config.get('train_csv')
    val_csv = data_config.get('val_csv')
    test_csv = data_config.get('test_csv')

    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # Initialize Datasets
    train_dataset = SkinLesionDataset(
        csv_path=train_csv,
        transform=train_transforms,
        apply_hair_removal=apply_hair_removal
    )
    
    val_dataset = SkinLesionDataset(
        csv_path=val_csv,
        transform=val_transforms,
        apply_hair_removal=apply_hair_removal
    )
    
    test_dataset = SkinLesionDataset(
        csv_path=test_csv,
        transform=val_transforms,
        apply_hair_removal=apply_hair_removal
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Dataloader Creation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"Successfully created dataloaders. Train batches: {len(train_loader)}")
    except Exception as e:
        print(f"Error: {e}")
