#!/usr/bin/env python
import pandas as pd
import rosbag_pandas
import sys, os
from PIL import Image
from io import BytesIO

sys.path.append(os.path.abspath('../..'))  # Not clean
from annotate.base import AnnotateBase

class BagReader(AnnotateBase):
    def __init__(self):
        super(BagReader, self).__init__()

    def _load_bag_data(self, file):
        bag = rosbag_pandas.bag_to_dataframe(file)
        bag = bag.rename(columns={'bebop_image_raw_throttle_compressed__data': 'data', 'bebop_image_raw_throttle_compressed__format': 'format'})

        df = bag[bag['format'].notnull()]
        self.image_data = df['data'].values
        self.num_images = self.image_data.size
        (self.width, self.height) = Image.open(BytesIO(self.image_data[0])).size

        assert self.width==856 and self.height==480, "Unexpected image dimensions (%d, %d)" % (self.width, self.height)
