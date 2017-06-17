#!/usr/bin/env python
#
from __future__ import print_function
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import argparse, sys, os, pdb

sys.path.append(os.path.abspath('..'))
from base import AnnotateBase
from shared.action import Action
from shared.imagewindow import ImageWindow
from shared.bagreader import BagReader


class Annotator(BagReader):
    def __init__(self):
        super(Annotator, self).__init__()

    def annotate(self, file):
        self._load_bag_data(file)
        w = self.width
        h = self.height
        s = self.scale
        c = self.chans
        size = w*h/s/s*c
        iw = ImageWindow(w, h)
        self.labels = np.empty(self.num_images, dtype='byte')
        self.data = np.empty((self.num_images, size), dtype='byte')

        # Check that our incoming image size is as expected...
        image = Image.open(BytesIO(self.image_data[0]))
        resized = image.resize((w/s, h/s), resample=Image.LANCZOS)
        hsv = resized.convert('HSV')
        assert size == np.fromstring(hsv.tobytes(), dtype='byte').size, "Unexpected image size!"

        i = 0
        while i < self.num_images:
            image = Image.open(BytesIO(self.image_data[i]))
            resized = image.resize((w/s, h/s), resample=Image.LANCZOS)
            hsv = resized.convert('HSV')
            #hue,_,_ = hsv.split()

            draw = ImageDraw.Draw(image)
            draw.line([(w/s, 0), (w/s, h)])
            draw.line([((s-1)*w/s, 0), ((s-1)*w/s, h)])
            # draw.line([(0, h/s), (w, h/s)])
            # draw.line([(0, (s-1)*h/s), (w, (s-1)*h/s)])

            iw.show_image(image)
            iw.force_focus()

            print('Image {} / {}:'.format(i, self.num_images), end='')

            iw.wait()

            key = iw.get_key()

            if key=='Escape':
                print('(QUIT)')
                break
            elif key=='BackSpace':
                if i > 0:
                    i -= 1
                print('(BACK)')
                continue
            elif key=='space':
                label = Action.SCAN
            elif key=='Return':
                label = Action.TARGET
            elif key=='Left':
                label = Action.TARGET_LEFT
            elif key=='Right':
                label = Action.TARGET_RIGHT
            elif self.num_actions > 4 and key=='Up':
                label = Action.TARGET_UP
            elif self.num_actions > 4 and key=='Down':
                label = Action.TARGET_DOWN
            else:
                label = Action.SCAN

            self.labels[i] = label
            self.data[i] = np.fromstring(hsv.tobytes(), dtype='byte')
            print(Action.name(label))
            i += 1

        iw.close()
        self.num_annotated = i


    def save(self, outfile):
        n = self.num_annotated
        np.savez(outfile, data=self.data[:n], labels=self.labels[:n])


    def convert(self, bagfile, npzfile_in, npzfile_out, mode):
        self._load_bag_data(bagfile)
        w = self.width
        h = self.height
        s = self.scale
        c = self.chans
        size = w*h/s/s*c

        assert mode in ['RGB', 'HSV', 'YCrCb']

        npz = np.load(npzfile_in)
        labels = self.labels = npz['labels']
        n = self.num_annotated = labels.shape[0]

        assert self.num_images == self.num_annotated

        data = self.data = np.empty((self.num_images, size), dtype='byte')

        for i in range(n):
            image = Image.open(BytesIO(self.image_data[i]))
            assert image.mode == 'RGB'
            resized = image.resize((w/s, h/s), resample=Image.LANCZOS)
            if mode=='RGB':
                new_img = resized
            else:
                new_img = resized.convert(mode)
            data[i] = np.fromstring(new_img.tobytes(), dtype='byte')
            print('{} : {}'.format(i, n))

        np.savez(npzfile_out, data=data, labels=labels)


def get_args():
    parser = argparse.ArgumentParser(description='Annotate drone flight images with action commands for supervised learning. NOTE: Python 2 required.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', metavar='bagfile_in', help='bagfile to analyze')
    parser.add_argument('outfile', metavar='npzfile_out', help='npz file for writing results')
    parser.add_argument('--convert', metavar='npzfile_in', help='npz file to convert')
    parser.add_argument('--mode', default='RGB', choices=['RGB', 'HSV', 'YCrCb'], help='Image format to which to convert')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    a = Annotator()

    if args.convert:
        a.convert(args.infile, args.outfile, args.convert, args.mode)
    else:
        a.annotate(args.infile)
        a.save(args.outfile)
