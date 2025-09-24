import cv2
import numpy as np
import os
from pathlib import Path

def remove_black_borders(image, threshold=5):
    """
    Removes black borders from an image.
    threshold: pixel intensity below which is considered 'black'
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask of non-black pixels
    mask = gray > threshold

    # Find coordinates of non-black pixels
    coords = np.argwhere(mask)

    if coords.size == 0:
        # Image is completely black
        return image

    # Bounding box of non-black pixels
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # +1 because slicing is exclusive

    # Crop the image
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def process_folder(input_folder, output_folder, threshold=5):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.*"):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = cv2.imread(str(file))
            if img is None:
                print(f"⚠️ Could not read {file}")
                continue

            cropped_img = remove_black_borders(img, threshold=threshold)

            save_path = output_path / file.name
            cv2.imwrite(str(save_path), cropped_img)
            print(f"✅ Processed: {file.name}")

if __name__ == "__main__":
    # Change these paths to your dataset
    source_folder = r"C:\Users\dangi\OneDrive\Desktop\THESIS\classified_images\not_bruised"
    output_folder = r"C:\Users\dangi\OneDrive\Desktop\THESIS\Latest(08_18_2025)\Datasets\test remove black borders\not_bruised"

    process_folder(source_folder, output_folder, threshold=5)