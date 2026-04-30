import cv2
import numpy as np

def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Remove hair from dermoscopic images using morphological black-hat filtering
    and TELEA inpainting.

    Args:
        image (np.ndarray): Input image in BGR format as a numpy array.

    Returns:
        np.ndarray: Cleaned image with hair removed in BGR format.
    """
    # Convert image to grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define kernel for morphological filtering (17x17 as specified)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    # Apply morphological black-hat filtering to isolate hair (dark structures on light background)
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)

    # Enhance hair features by thresholding
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image using the mask generated above using TELEA method
    inpainted_image = cv2.inpaint(image, thresh, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inpainted_image

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Hair removal for skin lesion images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned image")
    args = parser.parse_args()

    if os.path.exists(args.image):
        img = cv2.imread(args.image)
        if img is not None:
            clean_img = remove_hair(img)
            cv2.imwrite(args.output, clean_img)
            print(f"Cleaned image saved to {args.output}")
        else:
            print("Failed to load image.")
    else:
        print("Input image not found.")
