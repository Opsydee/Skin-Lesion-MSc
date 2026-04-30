import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet50 model for skin lesion classification.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        nn.Module: The configured ResNet50 model.
    """
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test ResNet50 Builder")
    parser.add_argument("--num_classes", type=int, default=7)
    args = parser.parse_args()

    model = build_resnet50(num_classes=args.num_classes)
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet50 built with {args.num_classes} classes.")
    print(f"Trainable params: {trainable_params} / {total_params}")
