#!/usr/bin/env python
#
import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
from PIL import Image as PILImage

import sys, os, pdb

sys.path.append(os.path.abspath('../..'))
from cnn.cnn import CNNModel
from shared.action import Action
from shared.imagewindow import ImageWindow


# TODO: For the love of Pete, get rid of these globals!

ns = '/bebop/'

iw = None
count = 0


# Publishers
takeoff  = rospy.Publisher(ns+'takeoff', Empty, latch=True, queue_size=1)
land     = rospy.Publisher(ns+'land', Empty, latch=True, queue_size=1)
reset    = rospy.Publisher(ns+'reset', Empty, latch=True, queue_size=1)
flattrim = rospy.Publisher(ns+'flattrim', Empty, latch=True, queue_size=1)
move     = rospy.Publisher(ns+'cmd_vel', Twist, latch=True, queue_size=1)

# Messages
right = Twist()
right.angular.z = -0.31415

left = Twist()
left.angular.z = 0.31415

forward = Twist()
forward.linear.x = 0.2

stop = Twist()

# Remember to scale commands for slow-mo!

def give_command(act, action):
    rospy.loginfo('Command {}'.format(action.name(act)))
    rospy.loginfo('-----')

    if act == action.SCAN or action.TARGET_RIGHT:
        command = right
    elif act == action.TARGET_LEFT:
        command = left
    elif act == action.TARGET:
        command = forward
    else:
        rospy.loginfo('Stop')
        command = stop

    move.publish(command)


def get_image():
    global iw, count

    msg = rospy.client.wait_for_message(ns+'image_raw', Image)
    h = msg.height
    w = msg.width
    s = 4  # hard-code proper size instead?


    if iw is None:
        iw = ImageWindow(w/s, h/s)

    rospy.loginfo('{}: Got {} x {} image'.format(count, w, h))
    count += 1

    image = PILImage.frombytes('RGB', (w, h), msg.data)
    resized = image.resize((w/s, h/s), resample=PILImage.LANCZOS)
    iw.show_image(resized).update()
    hsv = resized.convert('HSV')
    return np.fromstring(hsv.tobytes(), dtype='byte').reshape((h/s, w/s, 3))

# needed?
def image_callback(msg):
    rospy.loginfo('Got {} x {} image'.format(msg.height, msg.width))
    rospy.sleep(1)

def land_now():
    rospy.loginfo('land')
    land.publish()

def cnn_navigator():
    #CNN
    cnn = CNNModel(verbose=False)
    cnn.load_model()
    action = Action()

    # Inialize
    rospy.init_node('cnn_navigator')
    rospy.on_shutdown(land_now)

    # rospy.Subscriber(ns+'image_raw', Image, Image_callback, queue_size=1)
    # rospy.spin()

    rospy.loginfo('takeoff')
    # for i in range(5):
    #     takeoff.publish()
    #     rospy.sleep(.2)
    takeoff.publish()
    ### ADD THIS BACK IN!!!
    rospy.loginfo('MISSING SLEEP')
    #rospy.sleep(10)  # Would be better to get callback when ready...

    # rospy.loginfo('turn right')
    # move.publish(turn_right)
    # rospy.sleep(2)
    # for i in range(10):
    #     move.publish(turn_right)
    #     rospy.sleep(.2)

    # rospy.loginfo('stop')
    # move.publish(stop)
    #
    # rospy.loginfo('move forward')
    # move.publish(move_forward)
    # rospy.sleep(1)
    # for i in range(5):
    #     move.publish(move_forward)
    #     rospy.sleep(.2)
    #
    # rospy.loginfo('stop')
    # move.publish(stop)

    # rospy.sleep(3)
    # stop.publish()
    # rospy.sleep(10)
    # rospy.loginfo('land')
    # land.publish()


    # rate = rospy.Rate(4)
    # while not rospy.is_shutdown():
    #     msg = "hello world %s" % rospy.get_time()
    #     rospy.loginfo(msg)
    #     pub.publish(msg)
    #     rate.sleep()

    while not rospy.is_shutdown():
        img = get_image()
        act = cnn.predict_sample_class(img)  # could do predict-one_proba
        give_command(act, action)

    rospy.loginfo('land')
    land.publish()

if __name__ == '__main__':
    iw = None
    try:
        cnn_navigator()
    except rospy.ROSInterruptException:
        pass
