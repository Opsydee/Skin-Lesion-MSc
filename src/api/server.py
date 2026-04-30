import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import matplotlib.pyplot as plt

from src.models.efficientnet_b3 import build_efficientnet_b3
from src.preprocessing.augmentation import get_val_transforms
from src.xai.gradcam import generate_gradcam
from src.xai.lime_explainer import generate_lime
from src.xai.shap_explainer import generate_shap

app = FastAPI(title="Skin Lesion AI API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
val_transforms = get_val_transforms(224)

# Define classes (example list, adjust as per ISIC 2020 / HAM10000 mapping)
CLASS_NAMES = [
    "Actinic keratoses", 
    "Basal cell carcinoma", 
    "Benign keratosis-like lesions", 
    "Dermatofibroma", 
    "Melanoma", 
    "Melanocytic nevi", 
    "Vascular lesions"
]

@app.on_event("startup")
async def load_model():
    global model
    print("Loading model...")
    model = build_efficientnet_b3(num_classes=7, pretrained=True)
    
    checkpoint_path = "outputs/models/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No trained checkpoint found. Using ImageNet pre-trained weights for demonstration.")
        
    model = model.to(device)
    model.eval()

def array_to_base64(img_array: np.ndarray, colormap=None) -> str:
    """Convert numpy array to base64 encoded image string."""
    # Ensure values are 0-255 uint8
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
            
    if colormap is not None:
        img_array = cv2.applyColorMap(img_array, colormap)
        # Convert BGR to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    elif len(img_array.shape) == 2:
        # Convert grayscale to RGB for PIL
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(img_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay a heatmap on the original image."""
    if heatmap.dtype != np.uint8:
        heatmap = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # original_image should be RGB uint8
    overlayed = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
    return overlayed

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # For visualization, resize to 224x224 to match model output dimensions
    vis_image = cv2.resize(image_rgb, (224, 224))
    
    # Preprocess for model
    input_tensor = val_transforms(image_rgb).unsqueeze(0).to(device)
    
    # 1. Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
    pred_class_idx = int(pred_idx.item())
    confidence = float(conf.item()) * 100
    class_name = CLASS_NAMES[pred_class_idx]
    
    # 2. Grad-CAM
    # Target the last convolution layer of EfficientNet-B3
    target_layer = model.features[-1]
    gradcam_map = generate_gradcam(model, input_tensor, pred_class_idx, target_layer)
    gradcam_overlay = overlay_heatmap(vis_image, gradcam_map, alpha=0.6, colormap=cv2.COLORMAP_JET)
    gradcam_b64 = array_to_base64(gradcam_overlay)
    
    # 3. LIME
    # LIME requires a standard float image [0, 1]
    lime_input = vis_image.astype(np.float32) / 255.0
    lime_map = generate_lime(model, lime_input, pred_class_idx, num_samples=50) # Reduced samples for faster API response
    
    # Create LIME overlay: highlight positive superpixels in green
    lime_overlay = vis_image.copy()
    lime_mask = lime_map > 0
    lime_overlay[lime_mask, 0] = 0   # R
    lime_overlay[lime_mask, 1] = 255 # G
    lime_overlay[lime_mask, 2] = 0   # B
    # Blend with original
    lime_overlay = cv2.addWeighted(vis_image, 0.5, lime_overlay, 0.5, 0)
    lime_b64 = array_to_base64(lime_overlay)
    
    # 4. DeepSHAP
    # Load the true background samples if generated by train.py
    bg_path = "outputs/models/shap_background.pt"
    if os.path.exists(bg_path):
        # DeepExplainer is fast enough with 100 samples, but we can sample fewer for the API
        bg_samples = torch.load(bg_path, map_location=device)[:20] 
    else:
        # Fallback for demonstration
        print("Warning: shap_background.pt not found. Using random noise.")
        bg_samples = torch.randn(5, 3, 224, 224).to(device)
        
    try:
        shap_map = generate_shap(model, input_tensor, bg_samples)
        shap_overlay = overlay_heatmap(vis_image, shap_map, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
        shap_b64 = array_to_base64(shap_overlay)
    except Exception as e:
        print(f"SHAP Error: {e}")
        # Fallback if SHAP fails due to shape mismatch on dummy data
        shap_b64 = array_to_base64(vis_image)

    return {
        "class_name": class_name,
        "confidence": round(confidence, 1),
        "is_malignant": class_name in ["Melanoma", "Basal cell carcinoma"],
        "maps": {
            "gradcam": gradcam_b64,
            "lime": lime_b64,
            "shap": shap_b64
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
