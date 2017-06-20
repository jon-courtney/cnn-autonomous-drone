#!/usr/bin/env python
#
import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
from PIL import Image as PILImage
import math
import time

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
        self.forward.linear.x = 1.5/self.scale

        self.up = Twist()
        self.up.linear.z = 2/self.scale

        self.down = Twist()
        self.down.linear.z = -2/self.scale

        self.stop = Twist()


    def do(self, command, msg=None):

        assert command in ['takeoff', 'land', 'reset', 'flattrim', 'move']
        assert msg in [None, 'left', 'right', 'forward', 'stop', 'up', 'down']

        pub = self.__dict__[command]

        if command == 'move':
            nav_msg = self.__dict__[msg]
            pub.publish(nav_msg)
        else:
            pub.publish()

# TODO: Inherit from non-CNN navigator
# TODO: Add up() and down() methods

class CNNNavigator(object):

    def __init__(self, auto=False, display=False, speak=False, caution=False, verbose=False):
        self.auto = auto
        self.display = display
        self.speak = speak
        self.iw = None
        self._count = 0
        self.caution = caution
        self.forward = False
        self.forward_time = 0
        self.forward_margin = 2.0
        self.verbose = verbose

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
            self.loginfo('Loading CNN model...')
            self.cnn = CNNModel(verbose=False)
            self.cnn.load_model()
            self.loginfo('CNN model loaded.')
        else:
            self.cnn = None

        self.flying = False

    def _move_up(self, secs):
        self.loginfo('Going up...')
        if self.speak:
            self.speaker.speak('moving up')
        then = rospy.Time.now()
        d = rospy.Duration.from_sec(secs)
        r = rospy.Rate(10)
        while rospy.Time.now() - then < d:
            self.command.do('move', 'up')
            r.sleep()
        self.loginfo('Done.')


    def takeoff(self):
        self.logwarn('takeoff')
        if self.speak:
            self.speaker.speak("taking off")
        self.command.do('takeoff')
        self.flying = True
        self.loginfo('Waiting 10 seconds...')
        rospy.sleep(10)  # Would be better to get callback when ready...
        self._move_up(2.0)


    def land(self):
        self.loginfo('land')
        if self.speak:
            self.speaker.speak("landing")
        self.command.do('land')
        self.flying = False

    def flattrim(self):
        self.loginfo('flat trim')
        self.command.do('flattrim')

    def apply_caution(self, nav):
        if not self.caution:
            return nav

        if nav == 'forward':
            curr_time = time.time()
            if self.forward:
                if curr_time > self.forward_time + self.forward_margin:
                    # Apply the brakes
                    self.forward = False
                    nav = 'stop'
                    self.forward_time = curr_time
                    self.logwarn('Safety stop!')
                    if self.speak:
                        self.speaker.speak_new('safety stop')
            else:
                if curr_time > self.forward_time + self.forward_margin/2:
                    # ok, you can start again
                    self.forward = True
                    self.forward_time = curr_time
                else:
                    # still in time out
                    self.loginfo('In time out')
                    if self.speak:
                        self.speaker.speak_new('time out')
                    nav = 'stop'
        else:
            self.forward = False

        return nav

    def move(self, nav):
        assert nav in ['left', 'right', 'forward', 'stop']
        assert self.flying

        nav = self.apply_caution(nav)
        self.command.do('move', nav)
        self.loginfo(nav)


    def stop(self):
        self.move('stop')


    def loginfo(self, message):
        if self.verbose:
            rospy.loginfo(message)

    def logwarn(self, message):
        rospy.loginfo(message)  # There might be another method for this

    def handle_uncertainty(self, c, p):
        command = Action.name(c)
        if c == Action.SCAN and p < 0.5:
            self.logwarn('UNCERTAIN {} (p={:4.2f})'.format(command, p))
            if self.speak:
                self.speaker.speak_new('UNKNOWN')
            c = Action.TARGET_LEFT
        elif c == Action.TARGET and p < 0.5:
            self.logwarn('UNCERTAIN {} (p={:4.2f})'.format(command, p))
            if self.speak:
                self.speaker.speak_new('UNKNOWN')
            c = Action.TARGET_LEFT

        return c

    def navigate(self):
        assert self.auto
        assert self.flying

        self.loginfo('Begin autonomous navigation')
        try:
            while True:
                img = self.get_image()
                # pred = self.cnn.predict_sample_class(img)
                preds = self.cnn.predict_sample_proba(img)
                c = np.argmax(preds)
                p = preds[c]

                c = self.handle_uncertainty(c, p)

                self.give_command(c)
        except KeyboardInterrupt:
            self.loginfo('End autonomous navigation')
            self.cleanup()

    def watch(self):
        assert self.auto

        self.loginfo('Begin passive classification')
        try:
            while True:
                img = self.get_image()
                # pred = self.cnn.predict_sample_class(img)
                preds = self.cnn.predict_sample_proba(img)

                c = np.argmax(preds)
                p = preds[c]
                command = Action.name(c)

                c = self.handle_uncertainty(c, p)

                self.loginfo('Command {} (p={:4.2f})'.format(command, p))
                self.loginfo('-----')

                if self.speak:
                    self.speaker.speak_new(command)
        except KeyboardInterrupt:
            # It seems difficult to interrupt the loop when display=True
            self.loginfo('End passive classification')
            self.cleanup()


    def give_command(self, act):
        command = Action.name(act)
        self.loginfo('Command {}'.format(command))

        if self.speak:
            self.speaker.speak_new(command)

        if act in [Action.SCAN, Action.TARGET_RIGHT]:
            nav = 'right'
        elif act == Action.TARGET_LEFT:
            nav = 'left'
        elif act == Action.TARGET:
            nav = 'forward'
        else:
            self.loginfo('Stop')
            nav = 'stop'

        self.move(nav)
        self.loginfo('-----')


    def get_image(self):
        msg = rospy.client.wait_for_message(ns+'image_raw', Image)
        h = msg.height
        w = msg.width
        s = 4  # TODO: force expected size instead

        if self.display and self.iw is None:
            self.iw = ImageWindow(w/2, h/2)
            self._count = 0

        self.loginfo('{}: Got {} x {} image'.format(self._count, w, h))
        self._count += 1

        # Crop out the middle of the image
        # TODO: Tighten up the math for the processing loop
        image = PILImage.frombytes('RGB', (w, h), msg.data)\
                        .crop((w/s, h/s, (s-1)*w/s, (s-1)*h/s))
        resized = image.resize((w/s, h/s), resample=PILImage.LANCZOS)

        if self.display:
            self.iw.show_image(image).update()

        # Convert to HSV color space
        hsv = resized.convert('HSV')

        # Return properly-shaped raw data
        return np.fromstring(hsv.tobytes(), dtype='byte')\
                 .reshape((h/s, w/s, 3))

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
        # Just watching...
        # nav = CNNNavigator(auto=True, display=True, speak=False)
        # nav.watch()
        # nav.shutdown()

        # Yes, really flying
        nav = CNNNavigator(auto=True, verbose=True, speak=True, caution=False)
        nav.flattrim()
        nav.takeoff()
        nav.navigate()
        nav.land()
        nav.shutdown()
    except rospy.ROSInterruptException:
        nav.shutdown()
