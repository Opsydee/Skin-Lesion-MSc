# AGENTS.md — Skin Lesion Classification (MSc Project)
# Author: Ojedokun Opeyemi Emmanuel (252819)
# Supervisor: Dr Angela Makolo

## Project Overview
Build an end-to-end deep learning pipeline for multi-class skin lesion classification
using ResNet50 and EfficientNet-B3, with integrated Grad-CAM, LIME, and DeepSHAP
explainability methods evaluated against expert-annotated segmentation masks.

---

## Tech Stack
- Language: Python 3.10+
- Framework: PyTorch (preferred) with torchvision
- XAI: pytorch-grad-cam, lime, shap
- Image Processing: OpenCV (cv2), Pillow, scikit-image
- Data: NumPy, pandas
- Metrics: scikit-learn, torchmetrics
- Visualisation: matplotlib, seaborn
- Experiment Tracking: wandb (Weights & Biases)
- Version Control: Git

---

## Project Structure
```
skin_lesion_classification/
├── data/
│   ├── raw/              # Downloaded datasets (ISIC 2020, HAM10000, ISIC 2018)
│   ├── processed/        # Preprocessed and augmented images
│   └── splits/           # train/val/test CSV splits
├── src/
│   ├── preprocessing/
│   │   ├── hair_removal.py
│   │   ├── augmentation.py
│   │   └── class_balancing.py
│   ├── models/
│   │   ├── resnet50.py
│   │   ├── efficientnet_b3.py
│   │   └── train.py
│   ├── xai/
│   │   ├── gradcam.py
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── iou.py
│   │   └── pointing_game.py
│   └── utils/
│       ├── dataset.py
│       └── visualise.py
├── notebooks/            # Jupyter notebooks for exploration
├── configs/
│   └── config.yaml       # All hyperparameters
├── outputs/
│   ├── models/           # Saved checkpoints
│   ├── xai_maps/         # Attribution map images
│   └── results/          # Metric CSVs and plots
├── tests/
├── requirements.txt
└── README.md
```

---

## Coding Standards
- All functions must have docstrings (Google style)
- Use type hints throughout
- Fix random seeds for reproducibility: torch.manual_seed(42), np.random.seed(42)
- Log all hyperparameters to wandb or a config.yaml file
- Never hardcode paths — use config.yaml or argparse
- Save best model checkpoint based on validation AUC-ROC
- Use tqdm for all training/evaluation loops
- All scripts must be runnable from the command line with argparse

---

## Key Constants (use these defaults unless instructed otherwise)
- IMAGE_SIZE: 224x224
- BATCH_SIZE: 32
- LEARNING_RATE: 1e-4 (with cosine annealing)
- EPOCHS: 30
- TRAIN/VAL/TEST SPLIT: 70/15/15
- RANDOM_SEED: 42
- DEVICE: cuda if available else cpu

---

## Datasets
1. ISIC 2020 — primary classification training (33,126 images)
2. HAM10000 — multi-class training (10,015 images, 7 classes)
3. ISIC 2018 Task 1 — XAI ground truth (expert segmentation masks)

Download instructions: See README.md

---

## XAI Evaluation Protocol
- Threshold attribution maps at top 20% of pixel values to create binary masks
- Compute IoU between thresholded XAI mask and ISIC 2018 segmentation mask
- Compute Pointing Game: check if argmax(attribution_map) falls within segmentation mask
- Report mean IoU and Pointing Game accuracy per XAI method across the test set
- Compare all three methods (Grad-CAM, LIME, SHAP) under identical conditions

---

## Do NOT
- Use sudo for pip installs
- Hardcode dataset paths
- Skip docstrings or type hints
- Use random seeds without setting them explicitly
- Mix training and evaluation logic in the same script
