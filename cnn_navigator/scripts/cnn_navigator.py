#!/usr/bin/env python
#
# Revision $Id$

import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import pdb

ns = '/bebop/'

# Deal with command timeout parameter
# Remember to scale commands for slow-mo

def image_callback(msg):
    rospy.loginfo('Got {} x {} image'.format(msg.height, msg.width))
    rospy.sleep(1)

def cnn_navigator():
    # Add command for flattrim
    takeoff = rospy.Publisher(ns+'takeoff', Empty, latch=True, queue_size=1)
    land    = rospy.Publisher(ns+'land', Empty, latch=True, queue_size=1)
    move = rospy.Publisher(ns+'cmd_vel', Twist, latch=True, queue_size=1)

    turn_right = Twist()
    turn_right.angular.z = -0.31415

    move_forward = Twist()
    move_forward.linear.x = 0.2

    stop = Twist()

    rospy.init_node('cnn_navigator')


    # for i in range(60):
    #     msg = rospy.client.wait_for_message(ns+'image_raw', Image)
    #     rospy.loginfo('Got {} x {} image'.format(msg.height, msg.width))
    #     rospy.loginfo('Encoding: {}'.format(msg.encoding))
    #     rospy.loginfo('Data size: {}'.format(len(msg.data)))
    #     rospy.loginfo('Data type: {}'.format(type(msg.data)))
    #     rospy.sleep(1)

    # rospy.Subscriber(ns+'image_raw', Image, Image_callback, queue_size=1)
    # rospy.spin()

    rospy.loginfo('takeoff')
    # for i in range(5):
    #     takeoff.publish()
    #     rospy.sleep(.2)

    takeoff.publish()
    rospy.sleep(10)

    rospy.loginfo('turn right')
    move.publish(turn_right)
    rospy.sleep(2)
    # for i in range(10):
    #     move.publish(turn_right)
    #     rospy.sleep(.2)

    rospy.loginfo('stop')
    move.publish(stop)

    rospy.loginfo('move forward')
    move.publish(move_forward)
    rospy.sleep(1)
    # for i in range(5):
    #     move.publish(move_forward)
    #     rospy.sleep(.2)

    rospy.loginfo('stop')
    move.publish(stop)

    rospy.sleep(3)
    # stop.publish()
    # rospy.sleep(10)
    rospy.loginfo('land')
    land.publish()


    # rate = rospy.Rate(4)
    # while not rospy.is_shutdown():
    #     msg = "hello world %s" % rospy.get_time()
    #     rospy.loginfo(msg)
    #     pub.publish(msg)
    #     rate.sleep()

if __name__ == '__main__':
    try:
        cnn_navigator()
    except rospy.ROSInterruptException:
        pass
