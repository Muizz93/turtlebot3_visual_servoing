#!/usr/bin/env python 

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
import numpy as np
from numpy.linalg import inv
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
import yaml
import os


class pose_estimation:

    def __init__(self):
        self.image_pub = rospy.Publisher("/robot_detection", Image, queue_size=1)
        self.robot_pose_pub = rospy.Publisher("/robot_pose_raw", Float64MultiArray, queue_size=1)
        self.parking_pose_pub = rospy.Publisher("/parking_pose", Float64MultiArray, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.robot_pose = Float64MultiArray()
        self.parking_pose = Float64MultiArray()

        path = os.path.expanduser("~")+"/catkin_ws/src/turtlebot3_visual_servoing/camera_info/camera_sim.yaml"
        with open(path, "r") as file_handle:
            calib_data = yaml.load(file_handle)
        self.K = np.array(calib_data["camera_matrix"]["data"]).reshape(3,3)
        self.D = np.array(calib_data["distortion_coefficients"]["data"])

        goal_img = cv2.imread(os.path.expanduser("~")+'/catkin_ws/src/turtlebot3_visual_servoing/goal/desired.png')
        self.parking_hmat=[]
        parking_img, self.parking_hmat = self.compute_pose_estimation(goal_img, self.K, self.D)
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame_pose, robot_hmat = self.compute_pose_estimation(cv_image, self.K, self.D)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame_pose, "bgr8"))

            self.robot_pose.data = robot_hmat.flatten()
            
            matrix_dim = MultiArrayDimension()
            matrix_dim.label = "robot matrix"
            matrix_dim.size = robot_hmat.shape[0] * robot_hmat.shape[1]
            matrix_dim.stride = matrix_dim.size
            
            row_dim = MultiArrayDimension()
            row_dim.label = "row"
            row_dim.size = robot_hmat.shape[0]
            row_dim.stride = row_dim.size
            
            col_dim = MultiArrayDimension()
            col_dim.label = "column"
            col_dim.size = robot_hmat.shape[1]
            col_dim.stride = col_dim.size

            self.robot_pose.layout.dim = [matrix_dim, row_dim, col_dim]
            self.robot_pose.layout.data_offset = 0
            self.robot_pose_pub.publish(self.robot_pose)

            self.parking_pose.data = self.parking_hmat.flatten()
            
            matrix_dim.label = "parking matrix"
            matrix_dim.size = self.parking_hmat.shape[0] * self.parking_hmat.shape[1]
            matrix_dim.stride = matrix_dim.size

            row_dim.size = self.parking_hmat.shape[0]
            row_dim.stride = row_dim.size

            col_dim.size = self.parking_hmat.shape[1]
            col_dim.stride = col_dim.size 

            self.parking_pose.layout.dim = [matrix_dim, row_dim, col_dim]

            self.parking_pose.layout.data_offset = 0
            self.parking_pose_pub.publish(self.parking_pose)

        except CvBridgeError as e:
            print(e)


    def compute_pose_estimation(self, frame, matrix_coefficients, distortion_coefficients):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters_create()
        
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        # If markers are detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                ret = aruco.estimatePoseSingleMarkers(corners[i], 0.19, matrix_coefficients, distortion_coefficients)
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                # Draw a square around the markers
                aruco.drawDetectedMarkers(frame, corners) 

                # Draw Axis
                frame = aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)

        hmat = self.homogeneous_from_vectors(tvec, rvec)

        return frame, hmat

    def homogeneous_from_vectors(self, translation_vectors, rotation_vectors):
        rmat, _ = cv2.Rodrigues(rotation_vectors)
        hmat = np.r_['0,2', np.c_[rmat, translation_vectors.T], [0, 0, 0, 1]]
        return hmat

def main():
  rospy.init_node('pose_estimation', anonymous=True)
  rospy.loginfo("Initializing pose estimation")
  pose_estimation()
  rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
