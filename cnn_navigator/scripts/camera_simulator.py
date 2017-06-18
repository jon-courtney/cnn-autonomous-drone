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
sys.path.append(os.path.abspath('../..'))  # Not clean
from shared.imagewindow import ImageWindow
from shared.bagreader import BagReader

ns = '/bebop/'


class CameraSimulator(BagReader):
    def __init__(self, display=False):
        super(CameraSimulator, self).__init__()
        self.frame = 0
        self.msg = Image()
        self.camera = rospy.Publisher(ns+'image_raw', Image, latch=True, queue_size=1)
        self.display = display
        self.iw = None


    def next_image(self):
        i = self.frame % self.num_images
        image = PILImage.open(BytesIO(self.image_data[i]))

        # This logic should probably reside elsewhere
        if self.display:
            self.iw.show_image(image).update()

        self.frame += 1
        return image.tobytes()


    def make_image_msg(self, data):
        self.msg.height = self.height
        self.msg.width = self.width
        self.msg.encoding = 'rgb8'
        self.msg.step = len(data) / self.height
        self.msg.data = data
        return self.msg

    def log_image(self, img):
        rospy.loginfo('{}: {} x {} ({})'.format(self.frame, img.width, img.height, img.encoding))

    def simulate(self, file):
        self._load_bag_data(file)

        # This logic belongs elsewhere
        if self.display:
            self.iw = ImageWindow(self.width, self.height)

        # Initialize
        rospy.init_node('camera_simulator')

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            img_msg = self.make_image_msg(self.next_image())
            self.log_image(img_msg)
            self.camera.publish(img_msg)
            rate.sleep()

# TODO: Add argparse support?
if __name__ == '__main__':
    bagfile = 'test.bag'
    drone = CameraSimulator()
    try:
        drone.simulate(bagfile)
    except rospy.ROSInterruptException:
        pass
