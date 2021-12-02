#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class obstacle_avoidance:
    def __init__(self):
        self.image_pub = rospy.Publisher("/occupancy_image", Image, queue_size=1)
        # self.waypoints_pub = rospy.Publisher("/waypoints", Float64MultiArray, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            blurred = cv2.medianBlur(hsv, 25)

            # create a binary thresholded image on hue between red and yellow
            lower = (0,100,255)
            upper = (255,255,255)
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
                binary = cv2.drawContours(binary,[box],0,(0,0,0),cv2.FILLED)
            final = np.array(binary * 255, dtype = np.uint8)
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(final, "bgr8"))
    
        except CvBridgeError as e:
            print(e)

def main():
  rospy.init_node('obstacle_avoidance', anonymous=True)
  rospy.loginfo("Initializing obstacle avoidance")
  obstacle_avoidance()
  rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass