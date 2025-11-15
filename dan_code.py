import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def hsvColorSegmentation(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Error: image not found at path:", img_path)

    original = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -----------------------------------------
# 2. Adaptive Canny Thresholding (Otsu-based)
# -----------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_val = np.median(gray)

    lower_canny = int(max(0, 0.66 * median_val))
    upper_canny = int(min(255, 1.33 * median_val))

    edges = cv2.Canny(gray, lower_canny, upper_canny)

# -----------------------------------------
# 3. HSV Mango Segmentation (Adjustable)
# -----------------------------------------
    lowerBound = np.array([0, 50, 50])
    upperBound = np.array([150, 255, 255])

    mango_mask = cv2.inRange(img_hsv, lowerBound, upperBound)

# -----------------------------------------
# 4. Gap Mask (invert mango + threshold bottom half)
# -----------------------------------------
    gray2 = gray.copy()
    _, gap_mask = cv2.threshold(gray2, 80, 255, cv2.THRESH_BINARY_INV)

# -----------------------------------------
# 5. Find contours in mango mask
# -----------------------------------------
    mask_mango = cv2.absdiff(mango_mask, gap_mask)

    contours, _ = cv2.findContours(mask_mango, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find largest contour
    largest_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

# Create clean mango-only mask
    clean_mask = np.zeros_like(mask_mango)
    if largest_contour is not None:
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

# Draw bounding box
    bbox_img = original.copy()
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# -----------------------------------------
# 6. Visualization
# -----------------------------------------
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(mask_mango, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Mango Mask")
    plt.imshow(mango_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Gap Mask")
    plt.imshow(gap_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Bounding Boxes")
    plt.imshow(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Check if image path is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        print("Example: python script.py test.jpeg")
        sys.exit(1)
    
    # Get image path from command line argument
    imgPath = sys.argv[1]
    hsvColorSegmentation(imgPath)