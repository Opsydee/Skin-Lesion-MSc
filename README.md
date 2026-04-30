# Multi-Class Skin Lesion Classification and XAI Pipeline

This project is an end-to-end deep learning pipeline for multi-class skin lesion classification using ResNet50 and EfficientNet-B3. It integrates state-of-the-art Explainable AI (XAI) methods—Grad-CAM, LIME, and DeepSHAP—to provide transparency in medical image diagnostics. The pipeline evaluates the interpretability of these models against expert-annotated segmentation masks using Intersection over Union (IoU) and Pointing Game metrics, providing a standardized framework for evaluating clinical AI in dermatology.

## Datasets

The models and evaluations are built upon the following datasets. Please download and place them in `data/raw/`:

1. **ISIC 2020**: Primary classification training (33,126 images). Download from the [ISIC Archive](https://challenge2020.isic-archive.com/).
2. **HAM10000**: Multi-class training (10,015 images, 7 classes). Available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
3. **ISIC 2018 Task 1**: Expert segmentation masks used as ground truth for XAI evaluation. Download from [ISIC 2018 Challenge](https://challenge2018.isic-archive.com/).

Ensure your dataset splits are defined in `data/splits/train.csv`, `val.csv`, and `test.csv` before running the pipeline.

## Setup Instructions

1. **Clone the repository and enter the directory**:
   ```bash
   cd skin_lesion_classification
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment**:
   Update `configs/config.yaml` with your `wandb_entity`, dataset paths, and hyperparameters.

## How to Run

All scripts should be executed from the root directory. They use argparse for configurability (though defaults load from `config.yaml`).

### Training
To train the model:
```bash
python src/models/train.py --config configs/config.yaml
```

### XAI Generation
To generate attribution maps using a saved checkpoint:
```bash
python src/xai/gradcam.py --image path/to/image.jpg --checkpoint outputs/models/best_model.pth
python src/xai/lime_explainer.py --image path/to/image.jpg --checkpoint outputs/models/best_model.pth
python src/xai/shap_explainer.py --image path/to/image.jpg --checkpoint outputs/models/best_model.pth
```

### Evaluation
To evaluate standard classification metrics:
```bash
python src/evaluation/classification_metrics.py --checkpoint outputs/models/best_model.pth
```

To evaluate XAI maps (IoU and Pointing Game):
```bash
python src/evaluation/iou.py --map_dir outputs/xai_maps/ --mask_dir data/raw/isic2018_masks/
python src/evaluation/pointing_game.py --map_dir outputs/xai_maps/ --mask_dir data/raw/isic2018_masks/
```

## Expected Outputs

- **models/**: The best performing model checkpoint based on validation AUC-ROC will be saved in `outputs/models/`.
- **xai_maps/**: Visual attribution maps from Grad-CAM, LIME, and SHAP overlaying the original images will be exported to `outputs/xai_maps/`.
- **results/**: Classification performance metrics (Accuracy, Sensitivity, Specificity, F1, AUC-ROC) and XAI evaluation results (IoU, Pointing Game) will be saved as CSV files in `outputs/results/`.
- **wandb**: Real-time training and validation loss/metrics are logged directly to Weights & Biases.
