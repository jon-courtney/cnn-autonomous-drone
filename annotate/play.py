# annotate.py
#
import pandas as pd
import rosbag_pandas
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import argparse, sys, pdb

from ImageWindow import ImageWindow
from shared.Action import Action


class Annotator:
    def __init__(self):
        self.image_data = None
        self.num_images = 0
        self.width = 0
        self.height = 0
        self.scale = 4
        self.labels = None
        self.data = None
        self.num_annotated = 0

    def play(self, file):
        npz = np.load(file)
        data = self.data = npz['data']
        labels = self.labels = npz['labels']
        n = self.num_annotated = data.shape[0]
        w = self.width = 214
        h = self.height = 120
        size = w*h*3
        iw = ImageWindow(w, h)
        action = Action()

        assert size == data[0].size, "Buffer sizes do not match (%d vs %d)" % (size, data[0].size)

        i = 0
        while i < n:
            image = Image.frombytes('HSV', (w, h), data[i])
            iw.show_image(image)
            iw.force_focus()
            print 'Image %d / %d: %s' % (i, n, action.name(labels[i]))
            iw.wait()

            key = iw.get_key()

            if key=='Escape':
                break
            elif key=='BackSpace':
                if i > 0:
                    i -= 1
                continue
            else:
                i += 1

def get_args():
    parser = argparse.ArgumentParser(description='Play back drone flight images with associated action commands.  NOTE: Python 2 required.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', metavar='in_npzfile', help='npz file to play back')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    a = Annotator()
    a.play(args.infile)
