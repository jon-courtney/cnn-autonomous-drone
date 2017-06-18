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
from cnn_navigator import CNNNavigator
from shared.action import Action
from shared.speaker import Speaker


ns = '/bebop/'

class CNNLabelNavigator(CNNNavigator):

    def __init__(self, auto=False, display=False, speak=False):
        super(CNNLabelNavigator, self).__init__(auto=auto, display=display, speak=speak)
        self.total = 0
        self.correct = 0

    def get_label(self):
        msg = rospy.client.wait_for_message(ns+'label', String)
        return Action.value(msg.data)

    def watch(self):
        assert self.auto

        rospy.loginfo('Begin passive testing')
        try:
            while True:
                label = self.get_label()
                img = self.get_image()
                # pred = self.cnn.predict_sample_class(img)
                preds = self.cnn.predict_sample_proba(img)
                self.total +=1

                c = np.argmax(preds)
                p = preds[c]
                command = Action.name(c)

                if c != label:
                    rospy.loginfo('ERROR: predicted {}, expected {}'.format(command, Action.name(label)))
                else:
                    self.correct += 1

                acc = self.correct / float(self.total)

                if p < 0.5:
                    rospy.loginfo('UNCERTAIN {} (p={:4.2f})'.format(command, p))
                    if self.speak:
                        self.speaker.speak('UNKNOWN')
                    c = Action.SCAN
                    p = 0.0
                    command = Action.name(c)

                rospy.loginfo('Command {} (p={:4.2f}, acc={:6.4f})'.format(command, p, acc))
                rospy.loginfo('-----')

                if self.speak:
                    self.speaker.speak(command)
        except KeyboardInterrupt:
            rospy.loginfo('End passive testing')
            self.cleanup()

if __name__ == '__main__':
    try:
        nav = CNNLabelNavigator(auto=True, display=False, speak=False)
        nav.watch()
        nav.shutdown()
    except rospy.ROSInterruptException:
        nav.shutdown()
