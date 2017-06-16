#!/usr/bin/env python
#
import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import pandas as pd
import rosbag_pandas
import numpy as np

from PIL import Image as PILImage

import sys, os, pdb
sys.path.append(os.path.abspath('..'))
from base import AnnotateBase

ns = '/bebop/'



class DroneSimulator(AnnotateBase):
    def __init__(self):
        super(DroneSimulator, self).__init__()
        self.frame = 0
        self.msg = Image()
        self.camera  = rospy.Publisher(ns+'image_raw', Image, latch=True, queue_size=1)

    # Consider re-using from Annotator
    def _load_bag_data(file):
        bag = rosbag_pandas.bag_to_dataframe(file)
        bag = bag.rename(columns={'bebop_image_raw_throttle_compressed__data': 'data', 'bebop_image_raw_throttle_compressed__format': 'format'})

        df = bag[bag['format'].notnull()]
        self.image_data = df['data'].values
        self.num_images = self.image_data.size
        (self.width, self.height) = Image.open(BytesIO(self.image_data[0])).size

        assert self.width==856 and self.height==480, "Unexpected image dimensions (%d, %d)" % (self.width, self.height)

    def next_image(self):
        data = self.image_data[self.frame % self.num_images]
        self.frame += 1
        return data

    def image_from_data(self, data):
        self.msg.height = self.height
        self.msg.width = self.width
        self.msg.encoding = 'rgb8'
        self.msg.step = data.size / self.height
        self.msg.data = data
        return self.msg

    def log_image(self, img):
        rospy.loginfo('{}: {} x {} ({})'.format(self.frame, img.width, img.height, img.encoding))

    def simulate(self, file):
        self._load_bag_data(file)

        # Initialize
        rospy.init_node('drone_simulator')

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            img = self.image_from_data(self.next_image())
            self.log_image(img)
            self.camera.publish(img)
            rate.sleep()


if __name__ == '__main__':
    bagfile = 'test.bag'
    drone = DroneSimulator()
    try:
        drone.simulate(bagfile)
    except rospy.ROSInterruptException:
        pass
