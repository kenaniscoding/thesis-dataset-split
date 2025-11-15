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
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lowerBound = np.array([0,50,50])
    upperBound = np.array([150,255,255])
    mask = cv.inRange(hsv, lowerBound, upperBound)
    
    # Fix for OpenCV version compatibility
    # For OpenCV 3.x and 4.x, findContours returns 2 values
    # For OpenCV 2.x, it returns 3 values
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
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
    plt.title('HSV Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_contours)
    plt.title('Image with Contours')
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
