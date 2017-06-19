#!/usr/bin/env python
#
from __future__ import print_function
import numpy as np
import argparse, sys, os, pdb
from PIL import Image

sys.path.append(os.path.abspath('..'))


class Packer(object):
    def __init__(self):
        super(Packer, self).__init__()

    def read_jpegs(self, dir, num, label):
        w = 214
        h = 120
        c = 3
        size = w*h*c
        self.labels = np.zeros(num, dtype='byte')
        self.data = np.empty((num, size), dtype='byte')
        self.num_images = num

        # Check incoming images
        f = dir+'/'+str(1)+'.png'
        image = Image.open(f)
        assert (w, h) == image.size

        for i in range(1,num+1):
            f = dir+'/'+str(i)+'.png'
            image = Image.open(f)
            hsv = image.convert('HSV')
            self.labels[i-1] = label
            self.data[i-1] = np.fromstring(hsv.tobytes(), dtype='byte')

    def save(self, outfile):
        np.savez(outfile, data=self.data, labels=self.labels)

def get_args():
    parser = argparse.ArgumentParser(description='Combine directory of PNGs into npz file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', metavar='<dir>', help='directory with PNGs to pack')
    parser.add_argument('outfile', metavar='<npzfile_out>', help='npz file for writing results')
    parser.add_argument('num', metavar='num_jpegs', type=int, help='number of PNGs to read, starting with "1"')
    parser.add_argument('label', metavar='label', type=int, choices=[0, 1], help='label to apply to images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    p =  Packer()
    p.read_jpegs(args.dir, args.num, args.label)
    p.save(args.outfile)
