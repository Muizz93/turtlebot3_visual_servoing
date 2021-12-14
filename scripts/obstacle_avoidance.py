#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import heapq

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
            lower = (0,90,255)
            upper = (255,255,255)
            thresh = cv2.inRange(blurred, lower, upper)

            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
            # clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # get external contours
            contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            height, width, _ = img.shape
            # print(height,width)
            binary = np.zeros((height, width))
            for c in contours:
                cv2.drawContours(binary,[c],0,(255,255,255),cv2.FILLED)
                # get rotated rectangle from contour
                rot_rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)
                # draw rotated rectangle on copy of img
                binary = cv2.drawContours(binary,[box],0,(255,255,255),cv2.FILLED)
            final = np.array(binary, dtype = np.uint8)
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            a = np.uint8(final/255)
            b = np.zeros((a.shape[0]/128,a.shape[1]/128),dtype=np.uint8)
            for x in range(0, a.shape[0], 128):
                for y in range(0, a.shape[1], 128):
                    if np.any(a[x:x+128,y:y+128]):
                        b[int(x/128), int(y/128)]=1
                        break
            print(b)
            start = (7, 1)
            end = (1, 7)
            maze = np.array(b,dtype=np.uint8)
            path = astar(maze, start, end)
            print(path)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(final, "bgr8"))
    
        except CvBridgeError as e:
            print(e)





class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement = True):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :param allow_diagonal_movement: do we allow diagonal steps in our path
    :return:
    """
    

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    
    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 2

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1
        
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
                
        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            rospy.loginfo("giving up on pathfinding too many iterations")
            return return_path(current_node)

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []
        
        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            within_range_criteria = [
                node_position[0] > (len(maze) - 1),
                node_position[0] < 0,
                node_position[1] > (len(maze[len(maze) - 1]) - 1),
                node_position[1] < 0,
            ]
            
            if any(within_range_criteria):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                      ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            open_list.append(child)

def main():
  rospy.init_node('obstacle_avoidance')
  rospy.loginfo("Initializing obstacle avoidance")
  obstacle_avoidance()
  rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass