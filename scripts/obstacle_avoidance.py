#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class OccupancyMap:
    def __init__(self, occupancy_image, resolution=1):
        self.occupancy_map = occupancy_image
        self.resolution = resolution
        self.binary_occupancy_map = []

    def get_binary_occupancy_map(self):
        height, width, _ = self.occupancy_map.shape
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
    