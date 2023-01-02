#!/usr/bin/env python3

'''
The following code was developed by 2100386 (reg. no.) for the 2021-22
CE866 Computer Vision assignment.

Objective:
Identifying the position and orientation of a red pointer arrow when located on
top of a vintage London map.

Structure:
- Routines
- Main program
	- Map
	- Orientation
	- Pointer
	- Output

Description:
The first part of the code defines a set of routines used throughout
the assignment. The main program first segments the map (region of interest)
and rectifies the image. It then identifies the orientation arrow
to determine if the map is upside-down and rotates it if necessary.
Finally, the pointer arrow is processed to locate its tip and
calculate its bearing. These values are printed at the end.

Running the program:
The program was designed to identify a red arrow within a clear map
placed above a dark blue surface. To achieve good results the images
should match these specifications. You run it by passing exclusively
one argument in the command line, as demonstrated below.

	python3 mapreader.py <image-to-process>

Expected outputs:
When debugging mode is turned off, the program should print the following.

	POSITION x_value y_value
	BEARING angle

Debugging:
It is possible to print intermediate values and view images during processing
by seting DEBUGGING (below) to True.
'''

import sys
import numpy as np
import cv2 as cv
import math

DEBUGGING = False

#-------------------------------------------------------------------------------
# Routines.
#-------------------------------------------------------------------------------

def show_im(name, im):
	'''
	This routine opens an image in an pop-up window when in debugging mode.
	'''
	if DEBUGGING:
		cv.imshow(name, im)
		cv.waitKey(0)

def hsv_sat_thresh(im, sat_max, sat_min=0):
	'''
	This routine converts a BGR image to HSV and thresholds based on
	given saturation values. Saturation was chosen because the map has
	low saturation values, as opposed to the arrows and background.
	'''
	sat_max = convert_pctg(sat_max)
	sat_min = convert_pctg(sat_min)

	# The following four lines are adapted from
	# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

	# Convert
	hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
	low_range = (0, sat_min, 0)
	upper_range = (255, sat_max, 255)

	# Threshold
	return cv.inRange(hsv, low_range, upper_range)

def convert_pctg(x):
	'''
	This routine converts a percentage into a value ranging from 0 to 255.
	'''
	return int(x*255/100)

def blur_im(im,krnl):
	'''
	This routine reduces some noise with a Gaussian blur.
	Function adapted from https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
	'''
	return cv.GaussianBlur(im, krnl, 0)

def find_contours(im, arrows):
	'''
	This routine retrieves the external contours of a binary image.
	'''
	# Line adapted from https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
	contours, blob = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	
	# If we're looking for the arrows' contours, isolate them both.
	if arrows:
		contours = isolate_arrows(contours)

	return contours

def noise_red(im, krnl, arrows):
	'''
	This routine removes white noise in black background.
	'''
	# When cleaning arrows.
	if arrows:
		# Draw a black border to remove border noise.
		# (Line adapted from https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html)
		im = cv.rectangle(im, (0,0), (int(width), int(height)), (0,0,0), 40)
		show_im('Black border', im)
	
	# Remove white dots from the background.
	# (Line adapted from https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
	opened = cv.morphologyEx(im, cv.MORPH_OPEN, krnl)

	return opened

def rectify(im, corners):
	'''
	This routine sorts the map's corners, establishes the final position of the
	map, calculates the perspective transformation, and returns a rectified image.
	'''
	# The following four lines are adapted from
	# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	origCorners = sort_corners(corners)
	newCorners = np.array([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
	persp_mat = cv.getPerspectiveTransform(origCorners.astype(np.float32), newCorners.astype(np.float32))
	return cv.warpPerspective(im, persp_mat, (width, height))

def sort_corners(corners):
	'''
	This routine sorts the map's corners and returns an array in the order:
	Top Left, Top Right, Bottom Left, Bottom Right.
	'''
	sort = np.zeros((4,2))
	for corner in corners:
		x,y = corner.ravel()
		# top left
		if x < (width/2) and y < (height/2):
			sort[0] = corner
		# top right
		elif x > (width/2) and y < (height/2):
			sort[1] = corner
		# bottom left
		elif x < (width/2) and y > (height/2):
			sort[2] = corner
		# bottom right
		else:
			sort[3] = corner
	return sort

def isolate_arrows(conts):
	'''
	This routine returns a tuple with the pointer and orientation arrows.
	'''

	# The pointer arrow is the contour with the largest area
	pointer_cont = max(conts, key=cv.contourArea) 
	pointer_area = cv.contourArea(pointer_cont)

	# If more than 2 contours were detected, ignore them.
	for i, cont in enumerate(conts):
		A = cv.contourArea(cont) # Area of the contour
		if A == pointer_area: continue # This is the pointer arrow
		elif A < 1000: continue # These could be noise
		else: orientation_cont = cont # This is the orientation arrow

	return (pointer_cont, orientation_cont)

def rotate(im, contours, deg=180):
	'''
	This routine detects if the map is upside down and rotates it if necessary.
	'''
	# Get the orientation arrow's bounding rectangle.
	# (Lines adapted from https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
	x,y,w,h = cv.boundingRect(contours[1])
	if DEBUGGING:
		dirRect = im.copy()
		cv.rectangle(dirRect, (x,y), (x+w,y+h), (255,0,0), 2)
		show_im('Direction box', dirRect)

	# Calculate rectangle's centroid.
	cx, cy = x + w//2, y + h//2
	imcx, imcy = width/2, height/2

	# If upside-down...
	if cx < imcx and cy > imcy:
		
		if DEBUGGING:
			print('\nOriginally upside-down!')

		# Get the rotated image.
		# (Lines adapted from https://pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/)
		rotation_mat = cv.getRotationMatrix2D((imcx,imcy), deg, 1.0)
		finalIm = cv.warpAffine(im, rotation_mat, (width, height))

		# Convert to HSV and threshold.
		binary = hsv_sat_thresh(finalIm, 100, 30)
		show_im('HSV 3', binary)

		# Remove noise.
		kernel = np.ones((3,3), np.uint8)
		processed = noise_red(binary, kernel, arrows)
		show_im('Smooth 3', processed)

		# Find the arrows' contours.
		contours = find_contours(processed, arrows)
		if DEBUGGING:
			print('\nFound %d arrows.' % len(contours))
			# Draw the arrows' contours.
			arrowsIm = finalIm.copy()
			cv.drawContours(arrowsIm, contours, -1, (255, 0, 0), 2)
			show_im("Arrows' contours 2", arrowsIm)
		
	else:	
		finalIm = im
		if DEBUGGING: print('\nGood orientation!')
	
	return finalIm, contours[0]

def find_tip(corners):
	'''
	This routine finds and returns the tip of the pointer.
	'''
	corners = corners.reshape(3,2)
	
	# Empty array for the length between corners.
	lines = np.zeros((3,1))
	
	# Calculate the length between each of the two corners and save it.
	for i in range(3):
		lines[i] = line_length(corners[i-1], corners[i])
		# This way of indexing stores the distance between corners in the order:
		# (2 & 0), (0 & 1), and (1 & 2).
		# This is useful to identify the corners that make up the smallest line,
		# they will be: the corner with the same index and the previous one
		# (if index 2 is the smallest line, the corners that make it up are 2 and 1).
	if DEBUGGING:
		print('\nPointer lengths:')
		print(lines)
	
	# Identify the pointer's tip (opposite to the the smallest line).
	min_idx = np.argmin(lines)
	tip = corners[min_idx - 2]
	if DEBUGGING: print('\nTip:', tip)
	
	return tip[0], tip[1]

def line_length(c1, c2):
	'''
	This routine computes the Euclidean distance between two corners.
	'''
	x = abs(c1[0] - c2[0])
	y = abs(c1[1] - c2[1])

	return math.sqrt(x**2 + y**2)

def find_centroid(corners):
	'''
	This routine finds the centroid of a shape from a set of points.
	'''
	# The following three lines are taken from
	# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
	M = cv.moments(corners)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	
	return cx, cy

def calc_bearing(point, O):
	'''
	This routine calculates the bearing of the pointer's tip.
	'''

	# X and Y coordinates for tip (considering origin as the triangle's centroid).
	cx, cy = O[0], O[1]
	x = point[0] - cx
	y = cy - point[1]

	# Calculate the angle and convert to degrees.
	rad = math.atan2(y, x)
	deg = math.degrees(rad)

	# Convert to bearing.
	if deg < 0: bear = abs(deg) + 90.0
	elif deg > 0:
		if deg < 90: bear = 90 - deg
		elif deg > 90: bear = 360 - (deg - 90)
		else: bear = 0.0
	else: bear = 90.0

	return bear

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Ensure we were invoked with a single argument.

if len(sys.argv) != 2:
	print("Oops! No file to work on... Make sure you pass an image.")
	print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
	exit(1)
	
print("The filename to work on is %s." % sys.argv[1])

# Read in the image and get its shape. Notify if the file is invalid.
try:
	im = cv.imread(sys.argv[-1]) # (815, 1450, 3)
	height = im.shape[0]
	width = im.shape[1]
	show_im('Original', im)
except:
	print("\nOops! Invalid file... Make sure you pass an image.")
	print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
	exit(1)

# ------------------------------- MAP ------------------------------------------

# Not interested in the arrows at the moment.
arrows = False

# Convert to HSV and threshold based on saturation.
binary = hsv_sat_thresh(im, 40) # from 0% to 40% saturation
show_im('HSV', binary)

# Reduce noise.
kernel = (5,5)
blur = blur_im(binary, kernel)
show_im('Blurred', blur)

# Find the map's contour (external).
mapContour = find_contours(blur, arrows)

# Fill the binary image's contour.
# (Line adapted from https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python)
cv.drawContours(blur, mapContour, -1, (255, 255, 255), cv.FILLED)
show_im('Filled', blur)

# Smooth edges.
kernel = np.ones((5,5), np.uint8)
smooth = noise_red(blur, kernel, arrows)
show_im('Smooth', smooth)

# Finding and visualizing corners (returned as [X, Y] pairs).
# (Lines adapted from https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html)
mapCorners = cv.goodFeaturesToTrack(smooth, 4, 0.1, 400)

if DEBUGGING:
	circ = im.copy()
	for corner in mapCorners:
		x,y = corner.ravel()
		cv.circle(circ, (int(x),int(y)), 5, (0, 255, 0), -1)
	show_im('Map corners', circ)

# Rectify the map.
rectIm = rectify(im, mapCorners)
show_im('Rectified', rectIm)

# --------------------------- ORIENTATION --------------------------------------

# Now paying attention to the arrows.
arrows = True

# Find arrows (very low saturation to capture their tips better).
binary2 = hsv_sat_thresh(rectIm, 100, 30) # from 11% to 100% saturation
show_im('HSV 2', binary2)

# Remove noise.
kernel2 = np.ones((3,3), np.uint8) # 3x3 kernel to crop less of the pointer's tip
processed = noise_red(binary2, kernel2, arrows)
show_im('Smooth 2', processed)

# Find and visualize the arrows' contours (external).
arrowContours = find_contours(processed, arrows)

if DEBUGGING:
	print('\nFound %d arrows.' % len(arrowContours))
	# Draw the arrows' contours.
	arrowsIm = rectIm.copy()
	# (Following line adapted from https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
	cv.drawContours(arrowsIm, arrowContours, -1, (255, 0, 0), 2)
	show_im("Arrows' contours", arrowsIm)

# Get the final image (rotated, if upside down) and the pointer's contour.
finalIm, pointerContour = rotate(rectIm, arrowContours)
show_im("Final", finalIm)

# ----------------------------- POINTER ----------------------------------------

# Find and visualize the pointer's corners ([X, Y] pairs).
# (Following line adapted from http://amroamroamro.github.io/mexopencv/opencv/minarea_demo.html)
blob, triangleCorners = cv.minEnclosingTriangle(pointerContour)

if DEBUGGING:
	trian = finalIm.copy()
	for corner in triangleCorners:
		x,y = corner.ravel()
		cv.circle(trian, (int(x),int(y)), 5, (0, 255, 0), -1)
	show_im("Pointer corners", trian)

# Find and visualize the pointer's tip.
x, y = find_tip(triangleCorners)
tip = [x, y]

if DEBUGGING:
	tipIm = finalIm.copy()
	cv.circle(tipIm, (int(x),int(y)), 5, (0, 255, 0), -1)
	show_im("Tip", tipIm)

# Find and visualize the pointer's centroid.
cx, cy = find_centroid(triangleCorners)
cent = [cx, cy]

if DEBUGGING:
	centIm = finalIm.copy()
	cv.circle(centIm, (cx,cy), 5, (0, 0, 0), -1)
	show_im("Center", centIm)

# Calculate the pointer's bearing.
bear = calc_bearing(tip, cent)

# Output values.
xpos = x / width
ypos = (height - y) / height
hdg = bear

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------
# End of mapreader.py
#-------------------------------------------------------------------------------
