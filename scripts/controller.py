#!/usr/bin/env python

import rospy
import message_filters
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Pose2D
import numpy as np
from numpy.linalg import inv
import time
import tf

class Controller:
    def __init__(self, p_gain = 0.01, i_gain = 0, d_gain = 0, current_time = None, limitter = 0.3):
        self.kp, self.ki, self.kd = p_gain, i_gain, d_gain
        self.max_vel = limitter
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.robot_pose_sub = message_filters.Subscriber('robot_pose_raw', Float64MultiArray)
        self.parking_pose_sub = message_filters.Subscriber('parking_pose', Float64MultiArray)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.robot_pose_sub, self.parking_pose_sub], queue_size=1, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.robot_matrix)

        rospy.on_shutdown(self.fnShutDown)

        self.clear()
    
    def robot_matrix(self, robot_pose, parking_pose):
        self.robot_hmat = np.array(robot_pose.data).reshape(robot_pose.layout.dim[1].size, robot_pose.layout.dim[2].size)
        self.parking_hmat = np.array(parking_pose.data).reshape(parking_pose.layout.dim[1].size, parking_pose.layout.dim[2].size)

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

        self.robot_hmat = np.zeros((4,4), dtype=np.float64)
        self.parking_hmat = self.robot_hmat
        self.delta_hmat = self.robot_hmat
        self.theta = 0.0

    def update_controller(self, event=None):
        
        self.current_time = time.time()
        try:
            self.delta_hmat = np.matmul(inv(self.robot_hmat), self.parking_hmat)
        except np.linalg.LinAlgError:
            pass
        _,_,self.theta = tf.transformations.euler_from_matrix(self.delta_hmat, 'rxyz')
        self.err[2] = self.theta
        
        delta_time = self.current_time - self.last_time

        self.PTerm = self.kp * (self.err[2] - self.err[1])
        self.ITerm = self.ki * self.err[2] * delta_time
        self.DTerm = self.kd * (self.err[2] - 2*self.err[1] + self.err[0])

        if self.ITerm < -self.windup_guard:
            self.ITerm = -self.windup_guard
        elif self.ITerm > self.windup_guard:
            self.ITerm = self.windup_guard

        self.last_time = self.current_time
        self.err[0] = self.err[1]
        self.err[1] = self.err[2]
        print(self.err[2])

        self.u += self.PTerm + self.ITerm + self.DTerm

        self.u = np.clip(self.u, a_min = -self.max_vel, a_max = self.max_vel)
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.u

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
    
def main():
    rospy.init_node('controller', anonymous=True)
    rospy.loginfo("Initializing controller")
    goto_point = Controller(0.5, 0.00005, limitter = 2.84)
    rospy.Timer(rospy.Duration(1.0/10.0), goto_point.update_controller)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
