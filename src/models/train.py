import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from src.models.resnet50 import build_resnet50
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.utils.dataset import create_dataloaders
from src.preprocessing.class_balancing import compute_class_weights
from src.utils.shap_background import generate_and_save_background

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict, resume: bool = False) -> dict:
    """
    Train and validate the model.
    """
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Configuration extraction
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 30)
    lr = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 0.01)
    checkpoint_dir = training_config.get('checkpoint_dir', 'outputs/models/')
    num_classes = config.get('model', {}).get('num_classes', 7)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Calculate class weights if needed
    print("Computing class weights...")
    all_train_labels = [label for _, label, _ in train_loader.dataset]
    class_weights = compute_class_weights(all_train_labels).to(device)
    print(f"Class weights: {class_weights}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Metrics
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    auroc_metric = MulticlassAUROC(num_classes=num_classes).to(device)

    # Initialize wandb
    logging_config = config.get('logging', {})
    wandb.init(
        project=logging_config.get('wandb_project', 'skin_lesion_msc'),
        entity=logging_config.get('wandb_entity', None),
        config=config
    )
    wandb.watch(model, criterion, log="all", log_freq=10)

    start_epoch = 0
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    if resume:
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'best_val_auc' in checkpoint:
                best_val_auc = checkpoint['best_val_auc']
            print(f"Resumed at epoch {start_epoch} with Best Val AUC: {best_val_auc:.4f}")
        else:
            print("No latest_checkpoint.pth found. Starting from scratch.")

    print(f"Starting training on {device} from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        train_loss = train_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        accuracy_metric.reset()
        auroc_metric.reset()

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for images, labels, _ in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Update metrics
                preds = torch.softmax(outputs, dim=1)
                accuracy_metric.update(preds, labels)
                auroc_metric.update(preds, labels)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_metric.compute().item()
        val_auc = auroc_metric.compute().item()

        # Step Scheduler
        scheduler.step()

        # Logging
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'config': config
            }, checkpoint_path)
            print(f"Saved new best model with Val AUC: {best_val_auc:.4f}")

        # Save latest checkpoint for resuming
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'config': config
        }, latest_path)

    print("Training complete. Generating SHAP background tensor...")
    shap_bg_path = os.path.join(checkpoint_dir, "shap_background.pt")
    generate_and_save_background(train_loader, num_samples=100, save_path=shap_bg_path)

    wandb.finish()
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Skin Lesion Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Model
    model_config = config.get('model', {})
    arch = model_config.get('architecture', 'efficientnet_b3')
    num_classes = model_config.get('num_classes', 7)
    pretrained = model_config.get('pretrained', True)

    if arch == 'resnet50':
        model = build_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif arch == 'efficientnet_b3':
        model = build_efficientnet_b3(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Train
    train_model(model, train_loader, val_loader, config, resume=args.resume)
