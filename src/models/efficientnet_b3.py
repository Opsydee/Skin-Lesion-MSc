import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def build_efficientnet_b3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build an EfficientNet-B3 model for skin lesion classification.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        nn.Module: The configured EfficientNet-B3 model.
    """
    if pretrained:
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
    else:
        model = efficientnet_b3(weights=None)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 2 blocks in the features extractor
    # The features Sequential in EfficientNet-B3 has 8 main blocks plus a conv head
    # Let's unfreeze the last block (features[7]) and the conv head (features[8])
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.features[8].parameters():
        param.requires_grad = True

    # Replace the classifier head
    in_features = 1536 # EfficientNet-B3 output features before classifier
    
    # EfficientNet classifier is typically nn.Sequential(nn.Dropout, nn.Linear)
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test EfficientNet-B3 Builder")
    parser.add_argument("--num_classes", type=int, default=7)
    args = parser.parse_args()

    model = build_efficientnet_b3(num_classes=args.num_classes)
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B3 built with {args.num_classes} classes.")
    print(f"Trainable params: {trainable_params} / {total_params}")
