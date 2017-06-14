# annotate.py
#
import pandas as pd
import rosbag_pandas
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import argparse, sys, pdb

from imagewindow import ImageWindow
from shared.action import Action
from base import AnnotateBase


class Player(AnnotateBase):
    def __init__(self):
        super(Player, self).__init__()

    def play(self, file):
        npz = np.load(file)
        data = self.data = npz['data']
        labels = self.labels = npz['labels']
        n = self.num_annotated = data.shape[0]
        w = self.width = 214
        h = self.height = 120
        c = self.chans = 3
        size = w*h*c
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
    parser.add_argument('infile', metavar='npzfile_in', help='npz file to play back')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    a = Player()
    a.play(args.infile)
