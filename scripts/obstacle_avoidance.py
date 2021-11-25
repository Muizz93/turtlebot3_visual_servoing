#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2


class OccupancyMap:
    def __init__(self, raw_image, resolution=1):
        self.raw_image = raw_image
        self.resolution = resolution
        self.binary_occupancy_map = []

    def rectangle_detection(self):
        # convert to HSV, since red and yellow are the lowest hue colors and come before green
        hls = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2HLS)
        blurred = cv2.medianBlur(hls, 25)

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
        return np.array(binary)
        

    def get_binary_occupancy_map(self):
        detect = self.rectangle_detection()
        height, width, _ = detect.shape
        for i in range(height):
            for j in range(width):
                if self.occupancy_map > 128:
                    self.binary_occupancy_map[i][j] = True
                else:
                    self.binary_occupancy_map[i][j] = False 
        return self.binary_occupancy_map

class obstacle_avoidance:
    def __init__(self):
        self.occupancy_map_pub = rospy.Publisher("/binary_occupancy_image", Image, queue_size=1)
        self.waypoints_pub = rospy.Publisher("/waypoints", Float64MultiArray, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/image_raw/", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        map = OccupancyMap(cv_image, )