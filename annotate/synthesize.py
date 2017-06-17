#!/usr/bin/env python
#
from __future__ import print_function
import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np
import argparse, sys, os, pdb

sys.path.append(os.path.abspath('..'))
from base import AnnotateBase
from shared.action import Action


class Synthesizer(AnnotateBase):
    def __init__(self):
        super(Synthesizer, self).__init__()
        self.ops = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180]

    def transpose(self, image, op, label):
        simage = image.transpose(op)
        slabel = label

        # Only handling 4 actions right now
        assert label in [Action.SCAN, Action.TARGET, Action.TARGET_LEFT, Action.TARGET_RIGHT]

        if op in [Image.FLIP_LEFT_RIGHT, Image.ROTATE_180]:
            if label == Action.TARGET_LEFT:
                slabel = Action.TARGET_RIGHT
            elif label == Action.TARGET_RIGHT:
                slabel = Action.TARGET_LEFT

        return simage, slabel


    def synthesize(self, infile, outfile, mode='HSV'):
        npz = np.load(infile)
        data = self.data = npz['data']
        labels = self.labels = npz['labels']
        n = self.num_annotated = data.shape[0]
        w = self.width = 214
        h = self.height = 120
        c = self.chans = 3
        size = w*h*c
        ops = self.ops
        self.num_images = len(ops) * self.num_annotated
        outdata = np.empty((self.num_images, size), dtype='byte')
        outlabels = np.empty(self.num_images, dtype='byte')

        assert size == data[0].size, "Buffer sizes do not match (%d vs %d)" % (size, data[0].size)

        for i in range(n):
            image = Image.frombytes(mode, (w, h), data[i])
            print('Image {} / {}: {}'.format(i, n, Action.name(labels[i])))

            j = i * len(ops)
            for op in ops:
                simage, slabel = self.transpose(image, op, labels[i])
                outlabels[j] = slabel
                outdata[j] = np.fromstring(simage.tobytes(), dtype='byte')
                j += 1

        self.save(outfile, outdata, outlabels)


    def save(self, outfile, outdata, outlabels):
        np.savez(outfile, data=outdata, labels=outlabels)


def get_args():
    parser = argparse.ArgumentParser(description='Synthesize new data and action commands from existing data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', metavar='<npzfile_in>', help='npz file to read')
    parser.add_argument('outfile', metavar='<npzfile_out>', help='npz file to write with synthesized data')
    parser.add_argument('--mode', default='HSV', choices=['RGB', 'HSV', 'YCrCb'], help='Image format to use for reading and writing')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    s = Synthesizer()
    s.synthesize(args.infile, args.outfile, args.mode)
