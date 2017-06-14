# annotate.py
# Usage: ...
#
import pandas as pd
import rosbag_pandas
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import sys
from ImageWindow import ImageWindow
from Action import Action
import pdb


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

    def _load_data(self, file):
        bag = rosbag_pandas.bag_to_dataframe(file)
        bag = bag.rename(columns={'bebop_image_raw_throttle_compressed__data': 'data', 'bebop_image_raw_throttle_compressed__format': 'format'})

        df = bag[bag['format'].notnull()]
        self.image_data = df['data'].values
        self.num_images = self.image_data.size
        (self.width, self.height) = Image.open(BytesIO(self.image_data[0])).size

        assert self.width==856 and self.height==480, "Unexpected image dimensions (%d, %d)" % (self.width, self.height)

    def annotate(self, file):
        self._load_data(file)
        w = self.width
        h = self.height
        s = self.scale
        size = w*h/s/s*3
        iw = ImageWindow(w, h)
        self.labels = np.empty(self.num_images, dtype='byte')
        self.data = np.empty((self.num_images, size), dtype='byte')
        action = Action()

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

            print 'Image %d / %d:' % (i, self.num_images),

            iw.wait()

            key = iw.get_key()

            if key=='Escape':
                print '(QUIT)'
                break
            elif key=='BackSpace':
                if i > 0:
                    i -= 1
                print '(BACK)'
                continue
            elif key=='space':
                label = action.SCAN
            elif key=='Return':
                label = action.TARGET
            elif key=='Left':
                label = action.TARGET_LEFT
            elif key=='Right':
                label = action.TARGET_RIGHT
            elif key=='Up':
                label = action.TARGET_UP
            elif key=='Down':
                label = action.TARGET_DOWN
            else:
                label = action.SCAN

            self.labels[i] = label
            self.data[i] = np.fromstring(hsv.tobytes(), dtype='byte')
            print action.name(label)
            i += 1

        iw.close()
        self.num_annotated = i


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

    def save(self, outfile):
        n = self.num_annotated
        np.savez(outfile, data=self.data[:n], labels=self.labels[:n])

    def convert(self, bagfile, npzfile_in, npzfile_out, mode):
        self._load_data(bagfile)
        w = self.width
        h = self.height
        s = self.scale
        size = w*h/s/s*3

        npz = np.load(npzfile_in)
        labels = self.labels = npz['labels']
        n = self.num_annotated = labels.shape[0]

        assert self.num_images == self.num_annotated

        data = self.data = np.empty((self.num_images, size), dtype='byte')

        for i in range(n):
            image = Image.open(BytesIO(self.image_data[i]))
            resized = image.resize((w/s, h/s), resample=Image.LANCZOS)
            hsv = resized.convert(mode)
            data[i] = np.fromstring(hsv.tobytes(), dtype='byte')
            print "%d : %d" % (i, n)

        np.savez(npzfile_out, data=data, labels=labels)

    def convert_rgb(self, bagfile, npzfile_in, npzfile_out):
        self._load_data(bagfile)
        w = self.width
        h = self.height
        s = self.scale
        size = w*h/s/s*3

        npz = np.load(npzfile_in)
        labels = self.labels = npz['labels']
        n = self.num_annotated = labels.shape[0]

        assert self.num_images == self.num_annotated

        data = self.data = np.empty((self.num_images, size), dtype='byte')

        for i in range(n):
            image = Image.open(BytesIO(self.image_data[i]))
            rgb = image.resize((w/s, h/s), resample=Image.LANCZOS)
            assert rgb.mode == 'RGB'
            data[i] = np.fromstring(rgb.tobytes(), dtype='byte')
            print "%d : %d" % (i, n)

        np.savez(npzfile_out, data=data, labels=labels)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python annotate.py <bagfile> <npzfile>")
        sys.exit()

    if sys.argv[1] == '--play':
        a = Annotator()
        a.play(sys.argv[2])
    elif sys.argv[1] == '--hsv':
        if len(sys.argv) < 5:
            print("Usage: python annotate.py --hsv <bagfile> <npzfile_in> <npzfile_out>")
            sys.exit()
        a = Annotator()
        a.convert(sys.argv[2], sys.argv[3], sys.argv[4], 'HSV')
    elif sys.argv[1] == '--ycbcr':
        if len(sys.argv) < 5:
            print("Usage: python annotate.py --hsv <bagfile> <npzfile_in> <npzfile_out>")
            sys.exit()
        a = Annotator()
        a.convert(sys.argv[2], sys.argv[3], sys.argv[4], 'YCbCr')
    elif sys.argv[1] == '--rgb':
        if len(sys.argv) < 5:
            print("Usage: python annotate.py --rgb <bagfile> <npzfile_in> <npzfile_out>")
            sys.exit()
        a = Annotator()
        a.convert_rgb(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        a = Annotator()
        a.annotate(sys.argv[1])
        a.save(sys.argv[2])
