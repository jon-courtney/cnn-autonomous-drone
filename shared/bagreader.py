#!/usr/bin/env python
import pandas as pd
import rosbag_pandas
import sys, os, pdb
from PIL import Image
from io import BytesIO

sys.path.append(os.path.abspath('../..'))  # Not clean
from annotate_base import AnnotateBase

class BagReader(AnnotateBase):
    def __init__(self, num_actions=2, newtopic=True):
        super(BagReader, self).__init__(num_actions=num_actions)
        if newtopic:
            self.topic = 'bebop_image_raw_compressed_throttle'
        else:
            self.topic = 'bebop_image_raw_throttle_compressed'

    def _load_bag_data(self, file):
        bag = rosbag_pandas.bag_to_dataframe(file)
        bag = bag.rename(columns={self.topic+'__data': 'data', self.topic+'__format': 'format'})

        df = bag[bag['format'].notnull()]
        self.image_data = df['data'].values
        self.num_images = self.image_data.size
        (self.width, self.height) = Image.open(BytesIO(self.image_data[0])).size

        assert self.width==856 and self.height==480, "Unexpected image dimensions (%d, %d)" % (self.width, self.height)
