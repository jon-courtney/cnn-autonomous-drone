#!/usr/bin/env python
#
import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np

from PIL import Image as PILImage

from io import BytesIO
import sys, os, pdb

from camera_simulator import CameraSimulator
sys.path.append(os.path.abspath('../..'))  # Not clean
from shared.imagewindow import ImageWindow
from shared.bagreader import BagReader
from shared.action import Action

ns = '/bebop/'


class LabeledCameraSimulator(CameraSimulator):
    def __init__(self, display=False):
        super(LabeledCameraSimulator, self).__init__(display=display)
        self.label_topic = rospy.Publisher(ns+'label', String, latch=True, queue_size=1)
        self.label_msg = String()

    def next_labeled_image(self):
        i = self.frame % self.num_images
        label = self.labels[i]
        data = self.next_image() # self.frame is incremented here
        return data, label

    def make_label_msg(self, label):
        self.label_msg.data = Action.name(label)
        return self.label_msg

    def make_labeled_image_msgs(self, (data, label)):
        label_msg = self.make_label_msg(label)
        image_msg = self.make_image_msg(data)
        return image_msg, label_msg

    def log_labeled_image(self, img, label):
        rospy.loginfo('{}: {} x {} ({})'.format(self.frame, img.width, img.height, label))

    def simulate(self, bagfile, npzfile):
        self._load_bag_data(bagfile)
        npz = np.load(npzfile)
        labels = self.labels = npz['labels']

        # This logic belongs elsewhere
        if self.display:
            self.iw = ImageWindow(self.width, self.height)

        # Initialize
        rospy.init_node('labeled_camera_simulator')

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            img_msg, label_msg = self.make_labeled_image_msgs(self.next_labeled_image())
            self.log_labeled_image(img_msg, label_msg)
            self.camera.publish(img_msg)
            self.label_topic.publish(label_msg)
            rate.sleep()


# TODO: Add argparse support?
if __name__ == '__main__':
    bagfile = 'test.bag'
    npzfile = 'test.npz'
    drone = LabeledCameraSimulator()
    try:
        drone.simulate(bagfile, npzfile)
    except rospy.ROSInterruptException:
        pass
