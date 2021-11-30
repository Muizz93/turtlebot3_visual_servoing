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
# x = np.arange(0, height/10, width/10)
# y = np.arange(0, height/10, width/10)
# X, Y = np.meshgrid(x, y)
a = cv2.resize(binary,(width/20, height/20),interpolation=cv2.INTER_NEAREST)
# a = cv2.resize(a,(width, height),interpolation=cv2.INTER_NEAREST)
b = []
d = []
e = []
point=[1,1]
goal = [15,15]

point_float = [float(x) for x in point]
goal_float = [float(x) for x in goal]
# print(point_float)
dx = point_float[0]-goal_float[0]
dy = point_float[1]-goal_float[1]
while not np.hypot(dx,dy)<2.0 or np.hypot(dx,dy)>-2.0:
    # print('aaa')
    for i in range(3):
        # print(i)
        for j in range(3):
            if a[i-1+point[0]][j-1+point[1]] == True:
                
                c = np.hypot((i-1+point[0])-goal[0],(j-1+point[1])-goal[1])
                # print(c)
                b.append(c.tolist())
                # print(b)
                e = [(i-1+point[0]),(i-1+point[1])]
                # print(e)
                d.append(e)
    print(b)
    print(d)       
                
    # print(np.min(b))
    for k in range(len(b)):
        if np.min(b) == b[k]:
            point=d[k]
            point_float = [float(x) for x in point]
            print(point)
            dx = point_float[0]-goal_float[0]
            dy = point_float[1]-goal_float[1]
            break
    b=[]
    d=[]
    
    

#cv2.imshow("result1", result1)
cv2.imshow("result", binary)
cv2.imshow("resize", a)
cv2.waitKey(0)
cv2.destroyAllWindows()
