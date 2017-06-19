#!/usr/bin/env python
#
import rospy
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from io import BytesIO
import sys, os

# Import helper classes
sys.path.append(os.path.abspath('../..'))
from shared.imagewindow import ImageWindow
from shared.bagreader import BagReader

ns = '/bebop/'


class CameraSimulator(BagReader):
    def __init__(self, display=False, newtopic=True):
        # BagReader defaults assume num_actions=2 and newtopic=True
        super(CameraSimulator, self).__init__(newtopic=newtopic)
        self.frame = 0
        self.msg = Image()
        self.camera = rospy.Publisher(ns+'image_raw', Image, latch=True, queue_size=1)
        self.display = display
        self.iw = None


    def next_image(self):
        i = self.frame % self.num_images
        # Decode jpeg from previously-loaded bag data
        image = PILImage.open(BytesIO(self.image_data[i]))

        # This logic should probably reside elsewhere
        # Turn off display when in the processing loop
        if self.display:
            w = self.width
            h = self.height
            s = self.scale
            display_image = image.copy()
            draw = PILImageDraw.Draw(display_image)
            draw.rectangle([(w/s, h/s), ((s-1)*w/s, (s-1)*h/s)])
            self.iw.show_image(display_image).update()

        self.frame += 1

        # Return the decoded raw data
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
        # Load jpeg-encoded images from rosbag file
        self._load_bag_data(file)

        # This logic belongs elsewhere
        # Initialize display window
        if self.display:
            self.iw = ImageWindow(self.width, self.height)

        # Initialize ROS node
        rospy.init_node('camera_simulator')

        # Begin 4 Hz loop
        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            img_msg = self.make_image_msg(self.next_image())
            self.log_image(img_msg)
            self.camera.publish(img_msg)
            rate.sleep()

# TODO: Add argparse support?
if __name__ == '__main__':
    bagfile = 'test.bag' # Careful about what topics are used
    drone = CameraSimulator(display=False, newtopic=False)
    try:
        drone.simulate(bagfile)
    except rospy.ROSInterruptException:
        pass
