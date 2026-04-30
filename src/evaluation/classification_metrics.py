import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

def evaluate_model(model: nn.Module, test_loader: DataLoader, num_classes: int) -> dict:
    """
    Evaluate the classification model on standard metrics.
    Computes accuracy, per-class sensitivity/specificity, macro F1, and macro AUC-ROC.
    Saves results to a CSV file in outputs/results/.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        num_classes (int): Number of target classes.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Global Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Try computing AUC-ROC, depends on presence of all classes in test set
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        print("Warning: Could not compute AUC-ROC. Maybe not all classes present in test set.")
        macro_auc = float('nan')

    # Per-class sensitivity and specificity
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    sensitivity = []
    specificity = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivity.append(sens)
        specificity.append(spec)

    macro_sens = np.mean(sensitivity)
    macro_spec = np.mean(specificity)

    metrics = {
        'Accuracy': acc,
        'Macro_F1': macro_f1,
        'Macro_AUC': macro_auc,
        'Macro_Sensitivity': macro_sens,
        'Macro_Specificity': macro_spec
    }

    # Add per class metrics
    for i in range(num_classes):
        metrics[f'Class_{i}_Sensitivity'] = sensitivity[i]
        metrics[f'Class_{i}_Specificity'] = specificity[i]

    # Save to CSV
    os.makedirs('outputs/results', exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv('outputs/results/classification_metrics.csv', index=False)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    print(f"Results saved to outputs/results/classification_metrics.csv")

    return metrics

if __name__ == "__main__":
    import numpy as np
    # To run this directly requires a trained model and dataloader
    # In practice it is called within the training or testing script
    print("Classification Metrics Module Ready.")
