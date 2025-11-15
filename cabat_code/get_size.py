import argparse
import numpy as np
import cv2
from math import sqrt

PX_to_CM = 7.44 # cm

def get_size(image_frame, calibrated_pxm):
	#This function returns Y dimension, X dimension, Area, midx, midy, 
	gray = cv2.cvtColor(image_frame, 
						cv2.COLOR_BGR2GRAY)
	cv2.imwrite("gray.png", gray)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	cv2.imwrite("gaussianblur.png", gray)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	canny = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(canny, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	#Find contours
	cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#append contour into the area array
	areaArray = []
	for i, c in enumerate(cnts):
		area = cv2.contourArea(c)
		areaArray.append(area)


	sorteddata = sorted(zip(areaArray,cnts), key=lambda x: x[0], reverse=True)

	#largest Contour
	c = sorteddata[0][1] 
	x,y,w,h = cv2.boundingRect(c)
	
	#compute distance x and y
	dA = w
	dB = h

	# X,Y,Z parameters
	X = dA / calibrated_pxm
	Y = dB / calibrated_pxm

	# X = X * 25.4
	# Y = Y * 25.4

	if X > Y:
		Y_ = X
		X_ = Y
	else:
		Y_ = Y
		X_ = X

	return X_, Y_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image and size')
    parser.add_argument('image_path', help='Path to the image file')
    # parser.add_argument('size_value', type=float, help='Size value')
    
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from '{args.image_path}'!")
        exit(1)
        
    x, y = get_size(image, PX_to_CM)
    print(f"X Dimension (cm): {x:.2f}, Y Dimension (cm): {y:.2f}")
    # get_size(args.image_path, args.size_value)