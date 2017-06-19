#!/usr/bin/env python
#
from __future__ import print_function
import numpy as np
import argparse, sys, os, pdb

sys.path.append(os.path.abspath('..'))
from shared.bagreader import BagReader


class Extractor(BagReader):
    def __init__(self):
        super(Annotator, self).__init__()

    def extract(self, file):
        self._load_bag_data(file) # self.num_images set here

    def save(self, outfile):
        np.savez(outfile, img_data=self.image_data)


def get_args():
    parser = argparse.ArgumentParser(description='Extract drone flight images from ROS bag files. NOTE: Python 2 required.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', metavar='<bagfile_in>', help='bagfile to analyze')
    parser.add_argument('outfile', metavar='<npzfile_out>', help='npz file for writing results')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    e =  Extractor()
    e.extract(args.infile)
    e.save(args.outfile)
