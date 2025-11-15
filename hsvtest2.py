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
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # --- HSV to MASK ---
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lowerBound = np.array([0, 50, 50])
    upperBound = np.array([150, 255, 255])
    mask = cv.inRange(hsv, lowerBound, upperBound)

    # Find contours using combined edges
    contours, hierarchy = cv.findContours(
        combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # Create image with all contours drawn
    img_with_contours = imgRGB.copy()
    cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    mask_with_contours = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.drawContours(mask_with_contours, contours, -1, (0, 255, 0), 2)

    # ---- LARGEST CONTOUR ----
    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)
        mango_area_pixels = cv.contourArea(largest_contour)
        print(f"Largest Contour (Mango) Area: {mango_area_pixels:.2f} pixels")

        largest_contour_img = imgRGB.copy()
        cv.drawContours(largest_contour_img, [largest_contour], -1, (255, 0, 0), 4)

        # Show largest contour
        plt.figure(figsize=(6, 6))
        plt.imshow(largest_contour_img)
        plt.title("Largest Contour (Mango)")
        plt.axis('off')
        plt.show()

    else:
        print("No contours found!")

    # -------------------------------
    # Matplotlib Visualization
    # -------------------------------
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(imgRGB)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('HSV Mask')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(edged, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_with_contours)
    plt.title('Contours (Canny + HSV)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Show OpenCV windows
    cv.imshow('Mask with Contours', mask_with_contours)
    cv.imshow('Improved Contours', cv.cvtColor(img_with_contours, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Print contour information
    print(f"Found {len(contours)} contours")
    for i, contour in enumerate(contours):
        print(f"Contour {i}: {len(contour)} points, area: {cv.contourArea(contour):.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    imgPath = sys.argv[1]
    hsvColorSegmentation(imgPath)