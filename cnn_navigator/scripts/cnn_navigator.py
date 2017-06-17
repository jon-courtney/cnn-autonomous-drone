#!/usr/bin/env python
#
import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
from PIL import Image as PILImage
import math

import sys, os, pdb

sys.path.append(os.path.abspath('../..'))
from cnn.cnn import CNNModel
from shared.action import Action
from shared.imagewindow import ImageWindow


# TODO: For the love of Pete, get rid of these globals!

ns = '/bebop/'

iw = None
count = 0
display = False

# TODO: Make publishers static
class Command:
    def __init__(self, scale=10.0):
        self.scale = scale

        # Set up publishers
        self.takeoff  = rospy.Publisher(ns+'takeoff', Empty,
                                   latch=True, queue_size=1)
        self.land     = rospy.Publisher(ns+'land', Empty,
                                   latch=True, queue_size=1)
        self.reset    = rospy.Publisher(ns+'reset', Empty,
                                   latch=True, queue_size=1)
        self.flattrim = rospy.Publisher(ns+'flattrim', Empty,
                                   latch=True, queue_size=1)
        self.move     = rospy.Publisher(ns+'cmd_vel', Twist,
                                   latch=True, queue_size=1)

        # Set up messages
        self.right = Twist()
        self.right.angular.z = -math.pi/self.scale

        self.left = Twist()
        self.left.angular.z = math.pi/self.scale

        self.forward = Twist()
        self.forward.linear.x = 2/self.scale

        self.stop = Twist()


    def do(self, command, msg=None):

        assert command in ['takeoff', 'land', 'reset', 'flattrim', 'move']
        assert msg in [None, 'left', 'right', 'forward', 'stop']

        pub = self.__dict__[command]

        if command == 'move':
            nav_msg = self.__dict__[msg]
            pub.publish(nav_msg)
        else:
            pub.publish()


# TODO: Inherit from non-CNN navigator
# TODO: Add up() and down() methods

class CNNNavigator:

    def __init__(self, auto=False):
        self.auto = auto

        self.command = Command()

        # Initialize ROS node
        rospy.init_node('cnn_navigator')
        rospy.on_shutdown(lambda : self.emergency_land(self))

        if self.auto:
            # Load CNN
            rospy.loginfo('Loading CNN model...')
            self.cnn = CNNModel(verbose=False)
            self.cnn.load_model()
            rospy.loginfo('CNN model loaded.')
        else:
            self.cnn = None

        self.actions = Action()
        self.flying = False


    def takeoff(self):
        rospy.loginfo('takeoff')
        self.command.do('takeoff')
        self.flying = True
        rospy.loginfo('Waiting 10 seconds...')
        rospy.sleep(10)  # Would be better to get callback when ready...


    def land(self):
        rospy.loginfo('land')
        self.command.do('land')
        self.flying = False

    def flattrim(self):
        rospy.loginfo('flat trim')
        self.command.do('flattrim')

    def move(self, nav):
        assert nav in ['left', 'right', 'forward', 'stop']
        assert self.flying
        rospy.loginfo(nav)
        self.command.do('move', nav)


    def stop(self):
        self.move('stop')


    def navigate(self):
        assert self.auto
        assert self.flying

        rospy.loginfo('Begin autonomous navigation')
        while not rospy.is_shutdown():
            img = self.get_image()
            pred = self.cnn.predict_sample_class(img)  # could do predict-one_proba
            self.give_command(pred)
        rospy.loginfo('End autonomous navigation')

    def watch(self):
        assert self.auto

        rospy.loginfo('Begin passive classification')
        while not rospy.is_shutdown():
            img = self.get_image()
            pred = self.cnn.predict_sample_class(img)  # could do predict-one_proba
            rospy.loginfo('Command {}'.format(self.actions.name(pred)))
            rospy.loginfo('-----')
        rospy.loginfo('End passive classification')


    def give_command(self, act):
        rospy.loginfo('Command {}'.format(self.actions.name(act)))

        if act == self.actions.SCAN or self.actions.TARGET_RIGHT:
            nav = 'right'
        elif act == self.actions.TARGET_LEFT:
            nav = 'left'
        elif act == self.actions.TARGET:
            nav = 'forward'
        else:
            rospy.loginfo('Stop')
            nav = 'stop'

        self.move(nav)
        rospy.loginfo('-----')


    # TODO: remove globals
    def get_image(self):
        global iw, count, display

        msg = rospy.client.wait_for_message(ns+'image_raw', Image)
        h = msg.height
        w = msg.width
        s = 4  # TODO: hard-code expected size instead

        if display and iw is None:
            iw = ImageWindow(w/s, h/s)

        rospy.loginfo('{}: Got {} x {} image'.format(count, w, h))
        count += 1

        image = PILImage.frombytes('RGB', (w, h), msg.data)
        resized = image.resize((w/s, h/s), resample=PILImage.LANCZOS)

        if display:
            iw.show_image(resized).update()

        hsv = resized.convert('HSV')
        return np.fromstring(hsv.tobytes(), dtype='byte').reshape((h/s, w/s, 3))


    @staticmethod
    def emergency_land(obj):
        rospy.loginfo('emergency land')
        obj.command.do('land')


if __name__ == '__main__':
    try:
        nav = CNNNavigator(auto=True)
        nav.flattrim()
        nav.takeoff()
        nav.navigate()
        nav.land()
    except rospy.ROSInterruptException:
        pass
