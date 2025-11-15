import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def calculate_mango_area(image_path, gap_w_cm=3, gap_w_px=139):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mango color range (adjust as needed)
    lower = np.array([20, 40, 40])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No mango detected.")

    mango_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(mango_contour)

    # Use gap width only for scaling
    cm_per_pixel = gap_w_cm / gap_w_px

    mango_length_cm = w * cm_per_pixel  # horizontal axis
    mango_width_cm  = h * cm_per_pixel  # vertical axis
    mango_area_cm2  = mango_length_cm * mango_width_cm

    print(f"Mango length (horizontal): {mango_length_cm:.2f} cm")
    print(f"Mango width (vertical): {mango_width_cm:.2f} cm")
    print(f"Estimated Mango Area: {mango_area_cm2:.2f} cm²")

    # Visualization
    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask_clean, cmap="gray"); plt.title("Mask"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(output); plt.title("Bounding Box"); plt.axis("off")
    plt.show()

if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        print("Example: python script.py test.jpeg")
        sys.exit(1)
    
    # Get image path from command line argument
    imgPath = sys.argv[1]
    calculate_mango_area(imgPath, gap_w_cm=3, gap_w_px=139)