#!/usr/bin/env python
#
from __future__ import print_function
import numpy as np
import argparse, sys, os, pdb

sys.path.append(os.path.abspath('..'))
from shared.bagreader import BagReader

class Extractor(BagReader):
    def __init__(self, newtopic=True):
        super(Extractor, self).__init__(newtopic=newtopic)

    def extract(self, file):
        self._load_bag_data(file) # self.num_images set here

    def save(self, outfile):
        np.savez(outfile, image_data=self.image_data)

    def write_jpegs(self, prefix='bag-', num=-1):
        if num > self.num_images or num < 0:
            num = self.num_images
        for i in range(num):
            with open(prefix+'-'+str(i)+'.jpg', 'wb') as f:
                f.write(self.image_data[i])
            print('{} / {}'.format(i, num))

def get_args():
    parser = argparse.ArgumentParser(description='Extract drone flight images from ROS bag files. NOTE: Python 2 required.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', metavar='<bagfile_in>', help='bagfile to analyze')
    parser.add_argument('outfile', metavar='<npzfile_out>', help='npz file for writing results')
    parser.add_argument('--oldtopic', default=False, action='store_true', help='if set, forces use of /bebop/image_raw_throttle/compressed topic')
    parser.add_argument('--jpeg', default=False, action='store_true', help='write individual jpegs using <npzfile_out> as prefix')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    e =  Extractor(newtopic=not args.oldtopic)
    e.extract(args.infile)

    if(args.jpeg):
        e.write_jpegs(args.outfile)
    else:
        e.save(args.outfile)
