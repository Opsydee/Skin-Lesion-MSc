# ANTIGRAVITY AGENT MANAGER — TASK PROMPT
# Paste this into the Agent Manager when starting the project

---

## Task: Build MSc Skin Lesion Classification Pipeline

I am building an MSc research project on deep learning for skin lesion classification.
Read the AGENTS.md file in the project root first — it defines the full context,
tech stack, folder structure, and coding standards. Follow it exactly.

### Goal
Build a complete, reproducible end-to-end Python pipeline with the following 5 stages:

---

### Stage 1: Preprocessing Module

Build `src/preprocessing/`:

**hair_removal.py**
- Function: `remove_hair(image: np.ndarray) -> np.ndarray`
- Apply Black Hat morphological filtering (kernel size 17x17) to detect hair pixels
- Inpaint detected hair regions using `cv2.inpaint` with TELEA method
- Return cleaned image

**augmentation.py**
- Function: `get_train_transforms(image_size: int = 224) -> transforms.Compose`
- Include: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(30),
  ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
  RandomResizedCrop(224, scale=(0.8, 1.0)), ToTensor, Normalize(ImageNet mean/std)
- Function: `get_val_transforms(image_size: int = 224) -> transforms.Compose`
- Include: Resize(256), CenterCrop(224), ToTensor, Normalize(ImageNet mean/std)

**class_balancing.py**
- Function: `compute_class_weights(labels: list) -> torch.Tensor`
- Use sklearn `compute_class_weight` with strategy='balanced'
- Return weights as a torch.Tensor for use in CrossEntropyLoss

---

### Stage 2: Dataset Loader

Build `src/utils/dataset.py`:
- Class: `SkinLesionDataset(Dataset)`
- Accepts: csv_path (with columns: image_path, label), transform, apply_hair_removal flag
- Applies hair removal (if enabled) before transforms
- Returns: (image_tensor, label_int, image_path)
- Function: `create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]`
- Returns train, val, test DataLoaders using the config from config.yaml

---

### Stage 3: Model Module

Build `src/models/`:

**resnet50.py**
- Function: `build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module`
- Load torchvision ResNet50 with ImageNet weights
- Replace final FC layer with: Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)
- Freeze all layers except layer4 and the new classifier head

**efficientnet_b3.py**
- Function: `build_efficientnet_b3(num_classes: int, pretrained: bool = True) -> nn.Module`
- Load torchvision EfficientNet-B3 with ImageNet weights
- Replace classifier head with: Linear(1536, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)
- Freeze all layers except the last 2 blocks and the new classifier head

**train.py**
- Function: `train_model(model, train_loader, val_loader, config) -> dict`
- Use AdamW optimiser, CosineAnnealingLR scheduler
- Use CrossEntropyLoss with class weights
- Log train loss, val loss, val accuracy, val AUC-ROC each epoch to wandb
- Save best checkpoint (by val AUC-ROC) to outputs/models/
- Return dict with training history

---

### Stage 4: XAI Module

Build `src/xai/`:

**gradcam.py**
- Function: `generate_gradcam(model, image_tensor, target_class, target_layer) -> np.ndarray`
- Use pytorch-grad-cam library (GradCAM class)
- Return upsampled attribution map (same size as input image, values 0-1)

**lime_explainer.py**
- Function: `generate_lime(model, image_np, target_class, num_samples=1000) -> np.ndarray`
- Use lime.lime_image.LimeImageExplainer
- Return binary superpixel mask as a float attribution map (0 or 1 per superpixel)

**shap_explainer.py**
- Function: `generate_shap(model, image_tensor, background_tensor) -> np.ndarray`
- Use shap.DeepExplainer
- Return absolute SHAP values summed across colour channels, normalised 0-1

---

### Stage 5: Evaluation Module

Build `src/evaluation/`:

**iou.py**
- Function: `compute_iou(attribution_map: np.ndarray, seg_mask: np.ndarray, threshold: float = 0.2) -> float`
- Threshold attribution map at top `threshold` fraction of pixel values
- Compute IoU between thresholded binary map and ground truth binary segmentation mask

**pointing_game.py**
- Function: `compute_pointing_game(attribution_map: np.ndarray, seg_mask: np.ndarray) -> int`
- Return 1 if argmax of attribution_map falls inside seg_mask, else 0

**classification_metrics.py**
- Function: `evaluate_model(model, test_loader, num_classes) -> dict`
- Compute: accuracy, per-class sensitivity, specificity, macro F1, macro AUC-ROC
- Return dict and save results CSV to outputs/results/

---

### Config File

Create `configs/config.yaml`:
```yaml
data:
  image_size: 224
  batch_size: 32
  num_workers: 4
  train_csv: data/splits/train.csv
  val_csv: data/splits/val.csv
  test_csv: data/splits/test.csv
  apply_hair_removal: true

model:
  architecture: efficientnet_b3  # or resnet50
  num_classes: 7
  pretrained: true

training:
  epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.01
  seed: 42
  checkpoint_dir: outputs/models/

xai:
  gradcam_target_layer: features[-1]
  lime_num_samples: 1000
  shap_background_samples: 100
  attribution_threshold: 0.2

logging:
  wandb_project: skin_lesion_msc
  wandb_entity: your_username
```

---

### Requirements File

Create `requirements.txt` with all dependencies pinned.

---

### README.md

Create a README.md with:
- Project overview (1 paragraph)
- Dataset download instructions (ISIC 2020, HAM10000, ISIC 2018 Task 1)
- Setup instructions (virtualenv, pip install)
- How to run training, XAI, and evaluation scripts
- Expected outputs

---

### Acceptance Criteria
- [ ] All scripts run without errors from the command line
- [ ] Training loop logs metrics to wandb and saves best checkpoint
- [ ] All three XAI methods produce attribution maps of the same spatial size as input
- [ ] IoU and Pointing Game scores are computed and saved for each XAI method
- [ ] All functions have docstrings and type hints
- [ ] Random seed is fixed at 42 across torch, numpy, and random
- [ ] config.yaml controls all hyperparameters — nothing is hardcoded
