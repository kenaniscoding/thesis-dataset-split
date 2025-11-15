import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def calculate_mango_area(image_path):
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert BGR (OpenCV default) to RGB for matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Define HSV range for mango color (tweak as needed)
    lower = np.array([20, 40, 40])   # lower bound (H, S, V)
    upper = np.array([40, 255, 255]) # upper bound (H, S, V)

    # Step 4: Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Step 5: Morphological operations to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Step 6: Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No mango detected — adjust HSV range.")

    # Step 7: Select largest contour (assumed mango)
    mango_contour = max(contours, key=cv2.contourArea)

    # Step 8: Bounding rectangle
    x, y, w, h = cv2.boundingRect(mango_contour)

    # Step 9: Area calculation
    area = w * h

    # Step 10: Visualization with bounding box
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # Print results
    print(f"Length (height): {h} pixels")
    print(f"Width: {w} pixels")
    print(f"Estimated Mango Area (bounding box): {area} pixels²")

    # Step 11: Show results side by side
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(hsv)
    plt.title("HSV Conversion")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_clean, cmap="gray")
    plt.title("Thresholded Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(output_rgb)
    plt.title("Bounding Box")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        print("Example: python script.py test.jpeg")
        sys.exit(1)
    
    # Get image path from command line argument
    imgPath = sys.argv[1]
    calculate_mango_area(imgPath)