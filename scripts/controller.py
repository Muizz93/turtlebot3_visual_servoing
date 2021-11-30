#!/usr/bin/env python

import rospy
import message_filters
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Pose2D
import numpy as np
from numpy.linalg import inv
import time
import tf
import math

class Controller:
    def __init__(self): #, p_gain = 0.01, i_gain = 0, d_gain = 0, current_time = None, limitter = 0.3):
        # self.kp, self.ki, self.kd = p_gain, i_gain, d_gain
        # self.max_vel = limitter
        # self.current_time = current_time if current_time is not None else time.time()
        # self.last_time = self.current_time

        self.robot_pose_sub = message_filters.Subscriber('robot_pose_raw', Float64MultiArray)
        self.parking_pose_sub = message_filters.Subscriber('parking_pose', Float64MultiArray)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.robot_pose_sub, self.parking_pose_sub], queue_size=1, slop=1, allow_headerless=True)
        self.ts.registerCallback(self.robot_matrix)
        self.goto_point = PID(kp = np.array([[0.085, 0.0, 0.0], [0.0, 0.5, 0.0]]), 
                              ki = np.array([[0.0, 0.0, 0.0], [0.0, 0.001, 0.0]]), 
                              kd = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                              limitter = np.array([[0.2],[3.3]]))
        self.goto_parking = PID(kp = [[0.3, 0.8, -0.15], [0.0, 0.0, 0.0]])
        # self.clear()
        self.delta_hmat = np.zeros((4,4), dtype = np.float64)
        self.robot_hmat = np.zeros((4,4), dtype = np.float64)
        self.ds = 0.0
        self.theta = 0.0
        rospy.on_shutdown(self.fnShutDown)

    
    def robot_matrix(self, robot_pose, parking_pose):
        self.robot_hmat = np.array(robot_pose.data).reshape(robot_pose.layout.dim[1].size, robot_pose.layout.dim[2].size)
        self.parking_hmat = np.array(parking_pose.data).reshape(parking_pose.layout.dim[1].size, parking_pose.layout.dim[2].size)

    # def clear(self):

    #     """Clears PID computations and coefficients"""
            
    #     self.PTerm = 0.0
    #     self.ITerm = 0.0
    #     self.DTerm = 0.0
    #     self.err = [0.0, 0.0, 0.0]

    #     # Windup Guard
    #     self.int_error = 0.0
    #     self.windup_guard = 3.0

    #     self.u = 0.0

    #     self.robot_hmat = np.zeros((4,4), dtype=np.float64)
    #     self.parking_hmat = self.robot_hmat
    #     self.delta_hmat = self.robot_hmat
    #     self.theta = 0.0

    def update_controller(self, event=None):
        try:
            self.delta_hmat = np.matmul(inv(self.robot_hmat), self.parking_hmat)
        except np.linalg.LinAlgError:
            pass
        # _,_,self.theta = tf.transformations.euler_from_matrix(self.delta_hmat, 'rxyz')
        
        # print(self.delta_hmat)
        dx = self.delta_hmat[0][3]
        dy = self.delta_hmat[1][3]

        self.theta = np.arctan2(dy,dx)
        # self.ds = np.hypot(dx,dy)
        self.ds = math.hypot(dx,dy)
        print(self.ds,self.theta)

        # self.current_time = time.time()
        # delta_time = self.current_time - self.last_time

        # self.PTerm = self.kp * (self.err[2] - self.err[1])
        # self.ITerm = self.ki * self.err[2] * delta_time
        # self.DTerm = self.kd * (self.err[2] - 2*self.err[1] + self.err[0])

        # if self.ITerm < -self.windup_guard:
        #     self.ITerm = -self.windup_guard
        # elif self.ITerm > self.windup_guard:
        #     self.ITerm = self.windup_guard

        # self.last_time = self.current_time
        # self.err[0] = self.err[1]
        # self.err[1] = self.err[2]
        # print(self.err[2])

        # self.u += self.PTerm + self.ITerm + self.DTerm

        # u1 = 0.085* np.hypot(dx,dy)
        # if self.theta > 0.3 or self.theta < -0.3:
        #     u1 = 0
        # self.u = np.clip(self.u, a_min = -self.max_vel, a_max = self.max_vel)

        self.goto_point.update(errors = np.array([[self.ds], [self.theta], [0.0]]))
        if ((self.theta > 0.1) or (self.theta < -0.1)):
            self.goto_point.output[0] = 0
            print('a')

        # print('b')
        twist = Twist()
        twist.linear.x = self.goto_point.output[0]
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.goto_point.output[1]
        print(twist.linear.x,twist.angular.z)
        self.cmd_vel_pub.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

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

def near_parking(_ds):
    if _ds < 0.01 or _ds > -0.01:
        return True
    else:
        return False
    
def main():
    rospy.init_node('controller', anonymous=True)
    rospy.loginfo("Initializing controller")
    goto_point = Controller()
    rospy.Timer(rospy.Duration(1.0/10.0), goto_point.update_controller)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
