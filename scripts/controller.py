#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import numpy as np
from numpy.linalg import inv
import time

class Controller:
    def __init__(self, p_gain = 0.01, i_gain = 0, d_gain = 0, current_time = None):
        self.kp, self.ki, self.kd = p_gain, i_gain, d_gain
        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.robot_pose_sub = rospy.Subscriber("robot_pose_raw", Float64MultiArray, self.update_controller, queue_size = 1)
        self.parking_pose_sub = rospy.Subscriber("parking_pose", Float64MultiArray, self.parking_matrix)
        self.cmd_vel_pub = rospy.Publisher('/control/cmd_vel', Twist, queue_size = 1)

        rospy.on_shutdown(self.fnShutDown)

        self.clear()
    
    def parking_matrix(self, parking_pose):
        try:
            self.parking_hmat = np.array(parking_pose.data).reshape(parking_pose.layout.dim[1].size, parking_pose.layout.dim[2].size)

        except AttributeError:
            pass

    def clear(self):

        """Clears PID computations and coefficients"""

        self.SetPoint = 0.0
            
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.err = [0.0, 0.0, 0.0]

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 3.0

        self.u = 0.0

    def update_controller(self, robot_pose, current_time = None):
        try:
            self.robot_hmat = np.array(robot_pose.data).reshape(robot_pose.layout.dim[1].size, robot_pose.layout.dim[2].size)

            delta_hmat = np.matmul(inv(self.robot_hmat), self.parking_hmat)
            self.err[2] = delta_hmat[0][3]

            self.current_time = current_time if current_time is not None else time.time()
            delta_time = self.current_time - self.last_time

            self.PTerm = self.kp * (self.err[2] - self.err[1])
            self.ITerm = self.ki * self.err[2] * delta_time

            if self.ITerm < -self.windup_guard:
                self.ITerm = -self.windup_guard
            elif self.ITerm > self.windup_guard:
                self.ITerm = self.windup_guard

            self.DTerm = self.kd * (self.err[2] - 2*self.err[1] + self.err[0])

            self.last_time = self.current_time
            self.err[1] = self.err[2]
            self.err[0] = self.err[1]

            self.u += self.PTerm + self.ITerm + self.DTerm
            print(self.u)

        except AttributeError:
            print('a')
            pass

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
    
def main():
    rospy.init_node('controller', anonymous=True)
    rospy.loginfo("Initializing controller")
    goto_point = Controller(0.1)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
