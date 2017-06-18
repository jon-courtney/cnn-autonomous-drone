#!/usr/bin/env python
#
from __future__ import print_function
import pandas as pd
from PIL import Image
import numpy as np
import argparse, sys, os, pdb

sys.path.append(os.path.abspath('..'))
from shared.annotate_base import AnnotateBase
from shared.action import Action
from shared.imagewindow import ImageWindow


class Player(AnnotateBase):
    def __init__(self):
        super(Player, self).__init__()

    def play(self, file, mode='HSV'):
        npz = np.load(file)
        data = self.data = npz['data']
        labels = self.labels = npz['labels']
        n = self.num_annotated = data.shape[0]
        w = self.width = 214
        h = self.height = 120
        c = self.chans = 3
        size = w*h*c
        iw = ImageWindow(w, h)

        assert size == data[0].size, "Unexpected buffer size (%d vs %d)" % (size, data[0].size)

        assert data.shape[0] == labels.shape[0], "Mismatched image and label count"

        i = 0
        while i < n:
            image = Image.frombytes(mode, (w, h), data[i])
            iw.show_image(image)
            iw.force_focus()
            print('Image {} / {}: {}'.format(i, n, Action.name(labels[i])))
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
    parser.add_argument('--mode', default='HSV', choices=['RGB', 'HSV', 'YCrCb'], help='Image format to use for reading')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    a = Player()
    a.play(args.infile, args.mode)
