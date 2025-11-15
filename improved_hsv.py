import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def hsvColorSegmentation(imgPath):
    # Check if file exists
    if not os.path.isfile(imgPath):
        print(f"Error: File '{imgPath}' not found!")
        return
    
    img = cv.imread(imgPath)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from '{imgPath}'!")
        return

    # --------------------------------------------------------
    # ---------------- DSP IMPROVED PREPROCESSING ------------
    # --------------------------------------------------------

    # 1. Edge-preserving noise reduction
    filtered = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Convert to HSV
    hsv = cv.cvtColor(filtered, cv.COLOR_BGR2HSV)

# ---- Adaptive HSV Thresholding ----
    h, s, v = cv.split(hsv)

# Compute percentile-based adaptive thresholds
    H_min, H_max = np.percentile(h, [5, 95])
    S_min, S_max = np.percentile(s, [5, 98])
    V_min, V_max = np.percentile(v, [10, 98])

# Convert to uint8 bounds
    lowerBound = np.array([H_min, S_min, V_min], dtype=np.uint8)
    upperBound = np.array([H_max, S_max, V_max], dtype=np.uint8)

# Generate mask
    mask = cv.inRange(hsv, lowerBound, upperBound)
    
    # 3. HSV threshold (use your existing ranges)
    # lowerBound = np.array([0, 50, 50])
    # upperBound = np.array([150, 255, 255])
    # mask = cv.inRange(hsv, lowerBound, upperBound)

    # 4. Morphological smoothing (remove speckles and fill holes)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # 5. Edge enhancement via Canny
    # edges = cv.Canny(mask, 50, 150)

# 5. Adaptive Canny Edge Detection (median-based)
    gray = mask  # mask is already grayscale (0–255)

# Compute median pixel intensity
    v = np.median(gray)

# Adaptive thresholds based on image statistics
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

# Apply Canny with adaptive thresholds
    edges = cv.Canny(gray, lower, upper)

# Optional: preview window for debugging
    cv.imshow("Canny Edges", edges)
    
    # Combine mask with edges for sharper boundaries
    mask = cv.bitwise_and(mask, mask, mask=edges)

    # --------------------------------------------------------
    # ------------------ FIND CONTOURS -----------------------
    # --------------------------------------------------------

    # Use RETR_EXTERNAL to avoid internal noise contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Create a copy of the original image to draw contours on
    img_with_contours = imgRGB.copy()
    cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)

    # Also create a version with mask and contours
    mask_with_contours = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.drawContours(mask_with_contours, contours, -1, (0, 255, 0), 2)


    # ---- Find Largest Contour ----
    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)
        mango_area_pixels = cv.contourArea(largest_contour)
        print(f"Largest Contour (Mango) Area: {mango_area_pixels:.2f} pixels")

        # Create an RGB copy to draw on
        largest_contour_img = imgRGB.copy()

        # Draw ONLY the largest contour (blue)
        cv.drawContours(largest_contour_img, [largest_contour], -1, (0, 0, 255), 5)

        # Show using Matplotlib
        plt.figure(figsize=(6, 6))
        plt.imshow(largest_contour_img)
        plt.title("Largest Contour (Mango)")
        plt.axis('off')
        plt.show()
    else:
        print("No contours found!")


    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imgRGB)
    plt.title('Original Image (RGB)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Improved HSV Mask + Edges')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_contours)
    plt.title('Image with Improved Contours')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    cv.imshow('Mask with Contours', mask_with_contours)
    cv.imshow('Image with Contours', cv.cvtColor(img_with_contours, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Print contour information
    print(f"Found {len(contours)} contours")
    for i, contour in enumerate(contours):
        print(f"Contour {i}: {len(contour)} points, area: {cv.contourArea(contour):.2f}")

if __name__ == '__main__':
    # Check if image path is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        print("Example: python script.py test.jpeg")
        sys.exit(1)
    
    # Get image path from command line argument
    imgPath = sys.argv[1]
    hsvColorSegmentation(imgPath)