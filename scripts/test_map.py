#!/usr/bin/env python

import cv2
import numpy as np

img = cv2.imread("parking11.png")

# convert to HSV, since red and yellow are the lowest hue colors and come before green
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
blurred = cv2.medianBlur(hsv, 25)

# create a binary thresholded image on hue between red and yellow
lower = (0,100,255)
upper = (100,255,255)
thresh = cv2.inRange(blurred, lower, upper)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get external contours
contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
height, width, _ = img.shape
binary = np.ones((height, width))
for c in contours:
    cv2.drawContours(binary,[c],0,(255,255,255),cv2.FILLED)
    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    # draw rotated rectangle on copy of img
    cv2.drawContours(binary,[box],0,(0,0,0),cv2.FILLED)

#cv2.imshow("result1", result1)
cv2.imshow("result", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
