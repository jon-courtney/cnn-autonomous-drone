# hue-convert.py
# Usage: python annotate.py <bagfile>
#
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import sys


def convert_to_h(file):
    npz = np.load(file)
    indata = npz['data']
    labels = npz['labels']
    n = indata.shape[0]
    w = 214
    h = 120
    insize = w*h*3
    outsize = w*h
    outdata = np.empty((n, outsize), dtype='byte')
    outfile = file+'.hue'
    print "Filename: %s" % outfile

    for i in range(n):
        image = Image.frombytes('RGB', (w, h), indata[i])
        hsv = image.convert('HSV')
        hue,s,v = hsv.split()
        outdata[i] = np.fromstring(hue.tobytes(), dtype='byte')
        print "%d : %d" % (i, n)

    np.savez(outfile, data=outdata, labels=labels)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python hue-convert.py <npzfile>")
        sys.exit()

    convert_to_h(sys.argv[1])
