import torchvision.transforms as transforms

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get robust data augmentations for the training set.

    Args:
        image_size (int): Target size for the crop/resize operations. Default is 224.

    Returns:
        transforms.Compose: Composed transforms for training.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get standard transforms for the validation and test sets.

    Args:
        image_size (int): Target size for the final crop. Default is 224.

    Returns:
        transforms.Compose: Composed transforms for validation/testing.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
