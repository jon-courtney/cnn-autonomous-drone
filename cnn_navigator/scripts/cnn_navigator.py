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
from shared.speaker import Speaker


ns = '/bebop/'

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

    def __init__(self, auto=False, display=False, speak=False):
        self.auto = auto
        self.display = display
        self.iw = None
        self._count = 0
        self.speak = speak

        if speak:
            self.speaker = Speaker()
        else:
            self.speaker = None

        self.command = Command()

        # Initialize ROS node
        rospy.init_node('cnn_navigator', disable_signals=True)
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
        try:
            while True:
                img = self.get_image()
                # pred = self.cnn.predict_sample_class(img)
                preds = self.cnn.predict_sample_proba(img)
                c = np.argmax(preds)
                p = preds[c]

                if p < 0.5:
                    command = self.actions.name(c)
                    rospy.loginfo('UNCERTAIN {} (p={:4.2f})'.format(command, p))
                    if self.speak:
                        self.speaker.speak('UNKNOWN')
                    c = self.actions.value(self.actions.SCAN)
                    p = 0.0

                self.give_command(c)
        except KeyboardInterrupt:
            rospy.loginfo('End autonomous navigation')
            self.cleanup()

    def watch(self):
        assert self.auto

        rospy.loginfo('Begin passive classification')
        try:
            while True:
                img = self.get_image()
                # pred = self.cnn.predict_sample_class(img)
                preds = self.cnn.predict_sample_proba(img)

                c = np.argmax(preds)
                p = preds[c]

                if p < 0.5:
                    command = self.actions.name(c)
                    rospy.loginfo('UNCERTAIN {} (p={:4.2f})'.format(command, p))
                    if self.speak:
                        self.speaker.speak('UNKNOWN')
                    c = self.actions.value(self.actions.SCAN)
                    p = 0.0

                command = self.actions.name(c)

                rospy.loginfo('Command {} (p={:4.2f})'.format(command, p))
                rospy.loginfo('-----')

                if self.speak:
                    self.speaker.speak(command)
        except KeyboardInterrupt:
            rospy.loginfo('End passive classification')
            self.cleanup()


    def give_command(self, act):
        command = self.actions.name(act)
        rospy.loginfo('Command {}'.format(command))

        if self.speak:
            self.speaker.speak(command)

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


    def get_image(self):
        msg = rospy.client.wait_for_message(ns+'image_raw', Image)
        h = msg.height
        w = msg.width
        s = 4  # TODO: force expected size instead

        if self.display and self.iw is None:
            self.iw = ImageWindow(w/s, h/s)
            self._count = 0

        rospy.loginfo('{}: Got {} x {} image'.format(self._count, w, h))
        self._count += 1

        image = PILImage.frombytes('RGB', (w, h), msg.data)
        resized = image.resize((w/s, h/s), resample=PILImage.LANCZOS)

        if self.display:
            self.iw.show_image(resized).update()

        hsv = resized.convert('HSV')
        return np.fromstring(hsv.tobytes(), dtype='byte').reshape((h/s, w/s, 3))

    # TODO: Other functions here?
    def cleanup(self):
        if self.display and self.iw is not None:
            self.iw.close()
            self.iw = None

    def shutdown(self):
        self.cleanup()
        rospy.signal_shutdown('Node shutdown requested')

    @staticmethod
    def emergency_land(obj):
        rospy.loginfo('emergency land')
        obj.command.do('land')
        if obj.display and obj.iw is not None:
            obj.iw.close()
            obj.iw = None


if __name__ == '__main__':
    try:
        nav = CNNNavigator(auto=True, display=True, speak=False)
        nav.watch()
        nav.shutdown()

        # nav = CNNNavigator(auto=True)
        # nav.flattrim()
        # nav.takeoff()
        # nav.navigate()
        # nav.land()
        # nav.shutdown()
    except rospy.ROSInterruptException:
        nav.shutdown()
