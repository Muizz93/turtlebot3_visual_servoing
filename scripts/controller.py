#!/usr/bin/env python

import rospy
import message_filters
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Pose2D
import numpy as np
from numpy.linalg import inv
import time
import tf
from math import atan2, cos, sin, hypot

class Controller:
    def __init__(self): 

        self.robot_pose_sub = message_filters.Subscriber('robot_pose_raw', Float64MultiArray)
        self.parking_pose_sub = message_filters.Subscriber('parking_pose', Float64MultiArray)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.robot_pose_sub, self.parking_pose_sub], queue_size=1, slop=1, allow_headerless=True)
        self.ts.registerCallback(self.robot_matrix)
        self.goto_point = PID(kp = np.array([[0.095, 0.0, 0.0], [0.0, 0.5, 0.0]]), 
                              ki = np.array([[0.000000000001, 0.0, 0.0], [0.0, 0.0, 0.0]]), 
                              kd = np.array([[0.00001, 0.0, 0.0], [0.0, 0.001, 0.0]]),
                              limitter = np.array([[0.2],[2.6]]))
        self.goto_parking = PID(kp = [[0.3, 0.0, 0.0], [0.0, 0.8, -0.15]])
        # self.clear()
        self.delta_hmat = np.zeros((4,4), dtype = np.float64)
        self.robot_hmat = np.zeros((4,4), dtype = np.float64)
        self.ds = 0.0
        self.theta = 0.0
        rospy.on_shutdown(self.fnShutDown)

    def robot_matrix(self, robot_pose, parking_pose):
        self.robot_hmat = np.array(robot_pose.data).reshape(robot_pose.layout.dim[1].size, robot_pose.layout.dim[2].size)
        self.parking_hmat = np.array(parking_pose.data).reshape(parking_pose.layout.dim[1].size, parking_pose.layout.dim[2].size)

    def update_controller(self, event=None):
        try:
            self.delta_hmat = np.matmul(inv(self.robot_hmat), self.parking_hmat)
            _,_,thetag = tf.transformations.euler_from_matrix(self.parking_hmat, 'rxyz')
            # _,_,self.alpha = tf.transformations.euler_from_matrix(self.delta_hmat, 'rxyz')
            _,_,theta = tf.transformations.euler_from_matrix(self.robot_hmat, 'rxyz')
            # print(self.delta_hmat)
        except np.linalg.LinAlgError:
            pass
        # 
        
        # print(self.delta_hmat)
        dx = self.delta_hmat[0][3]
        dy = self.delta_hmat[1][3]

        self.angle = atan2(dy,dx)
        
        self.alpha = self.normalize(self.angle - theta)
        self.beta = self.normalize(thetag - self.angle)
        self.rho = hypot(dy, dx)
        # print(self.ds,self.theta)
        # print(self.theta,self.alpha)

        u = np.zeros(2, dtype=np.float64)

        if (not self.near_parking()):

            self.goto_point.update(errors = np.array([[self.rho], [self.angle], [0.0]]))
            if ((self.angle > 0.1) or (self.angle < -0.1)):
                u[0] = 0.0
            else:
                u[0] = self.goto_point.output[0]
            u[1] = self.goto_point.output[1]
            # print('b')
            twist = Twist()
            twist.linear.x = u[0]
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = u[1]
            self.cmd_vel_pub.publish(twist)
            # print(twist.linear.x, twist.angular.z)
        else:
            rospy.loginfo_once("Near the Parking")
            # if not self.at_the_parking():
            self.goto_parking.update(errors = np.array([[self.rho], [self.alpha], [self.beta]]))
            twist = Twist()
            twist.linear.x = self.goto_parking.output[0]
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.goto_parking.output[1]
            self.cmd_vel_pub.publish(twist)
            # else:
                # self.fnShutDown()
    def normalize(self, angle):
        return atan2(sin(angle), cos(angle))

    def near_parking(self):
        if self.rho < 0.1:# or self.rho > -0.1:
            return True
        else:
            return False

    def at_the_parking(self):
        if (self.rho < 0.005) and ((self.beta -self.alpha) < 0.01 or (self.beta -self.alpha)< -0.01):
            rospy.loginfo_once("Reached the Parking")
            return True
        else:
            return False

    def fnShutDown(self):
        rospy.loginfo_once("Shutting down. cmd_vel will be 0")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        

class PID:
    def __init__(self, kp = np.zeros((2,3),dtype=np.float64), ki = np.zeros((2,3),np.float64), kd = np.zeros((2,3),dtype=np.float64), current_time = time.time(), limitter = np.array([[0.3],[0.3]])):
        self.kp, self.ki, self.kd= kp, ki, kd
        self.current_time = current_time
        self.limitter = limitter
        self.windup_guard = np.array([[20.0],[20.0]])
        
        self.clear()

    def update(self, errors, current_time = time.time()):
        self.current_errors = errors
        self.current_time = current_time    
        self.delta_time = self.current_time - self.previous_time
        # print(self.current_errors.shape)
        self.PTerm = np.matmul(self.kp, (self.current_errors - self.previous_errors))
        self.ITerm = np.matmul(self.ki, (self.current_errors * self.delta_time))
        self.DTerm = np.matmul(self.kd, (self.current_errors - 2*self.previous_errors + self.last_errors))
        # print(self.DTerm)
        if np.any(self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif np.any(self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard
        
        self.previous_time = self.current_time
        self.last_errors = self.previous_errors
        self.previous_errors = self.current_errors

        self.output += self.PTerm + self.ITerm + self.DTerm
        self.output[0] = np.clip(self.output[0], a_min = -self.limitter[0], a_max = self.limitter[0])
        self.output[1] = np.clip(self.output[1], a_min = -self.limitter[1], a_max = self.limitter[1])
        # print(self.output)
    def clear(self):
        self.PTerm = np.zeros((2,1),dtype=np.float64)
        self.ITerm = np.zeros((2,1),dtype=np.float64)
        self.DTerm = np.zeros((2,1),dtype=np.float64)
        self.current_errors = np.zeros((3,1),dtype=np.float64)
        self.previous_errors = np.zeros((3,1),dtype=np.float64)
        self.last_errors = np.zeros((3,1),dtype=np.float64)
        self.previous_time = 0.0
        self.delta_time = 0.0
        self.output = np.zeros((2,1),dtype=np.float64)


    
def main():
    rospy.init_node('controller', anonymous=True)
    rospy.loginfo("Initializing controller")
    goto_point = Controller()
    rospy.Timer(rospy.Duration(1.0/20.0), goto_point.update_controller)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
