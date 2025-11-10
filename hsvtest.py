import cv2

# Read the image
fr_img = cv2.imread('sample_green.png')
bg_img = cv2.imread('background1.png')

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
fgMask = cv2.absdiff(resized_hsv, bg_resized_hsv)
cv2.imshow('Foreground Mask', fgMask)
cv2.waitKey(0)
gray = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow('Grayscale with Blur Edges', gray)
cv2.waitKey(0)
edged = cv2.Canny(gray, 80, 100)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
