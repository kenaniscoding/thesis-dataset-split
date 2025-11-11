import cv2, numpy as np

# Read the image
# fr_img = cv2.imread('paper_dataset/mango_3_top.png')
fr_img = cv2.imread('paper_dataset/mango_3_bottom.png')
bg_img = cv2.imread('paper_dataset/mango_3_background.png')

# Convert to HSV
hsv_img = cv2.cvtColor(fr_img, cv2.COLOR_BGR2HSV)
hsv_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

# Decrease scale (e.g., 50% smaller)
scale = 0.5
new_width = int(hsv_img.shape[1] * scale)
new_height = int(hsv_img.shape[0] * scale)
other_width = int(hsv_bg_img.shape[1] * scale)
other_height = int(hsv_bg_img.shape[0] * scale)

# Resize the HSV image
resized_hsv = cv2.resize(hsv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
bg_resized_hsv = cv2.resize(hsv_bg_img, (other_width, other_height), interpolation=cv2.INTER_AREA)

# Display the resized HSV image
cv2.imshow('Resized HSV Image', resized_hsv)
cv2.waitKey(0) 
cv2.imshow('Bg HSV Image', bg_resized_hsv)
cv2.waitKey(0) 

# Foreground Mask Creation using Absolute Difference
fgMask = cv2.absdiff(resized_hsv, bg_resized_hsv)
cv2.imshow('Foreground Mask', fgMask)
cv2.waitKey(0)
gray = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

# Attempted Gaussian and Bilateral Blur but Median Blur worked best
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
# cv2.imshow('Gaussian Blurred', gray)
# cv2.waitKey(0)

median = cv2.medianBlur(gray, 7)
cv2.imshow('Median Blurred', median)
cv2.waitKey(0)

# bilateral = cv2.bilateralFilter(gray, 15, 150, 150)  
# cv2.imshow('Bilateral Blur', bilateral)
# cv2.waitKey(0)

# Apply automatic Canny thresholding
# 0.66 and 1.33 come from a common heuristic used in the “auto-Canny” method
# typical 0.66 and 1.33
# conservative 0.5 and 1.5
# aggressive 0.7 and 1.3
v = np.median(gray)
lower = int(max(0, 0.6 * v))
upper = int(min(255, 1.33 * v))

# Sobel edge detection
# tried but Canny worked better
edged = cv2.Canny(gray, lower, upper)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)

# Find contours and get the largest one
contours, hierarchy = cv2.findContours(edged,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_contour = max(contours, key=cv2.contourArea)

# Draw a bounding rectangle around the mango
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw the rectangle for visualization
output = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Bounding Box', output)
cv2.waitKey(0)

# Print measurements
print(f"Width (pixels): {w}")
print(f"Height (pixels): {h}")

cv2.destroyAllWindows()