# concat.py
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python {} <npzfile1> <npzfile2> ... <npzfileN> <npzoutfile>".format(sys.argv[0]))
        sys.exit()

    files = sys.argv[1:-1]
    outfile = sys.argv[-1]
    print('Concatenating {}'.format(files))
    print('Outfile: {}'.format(outfile))
    n = len(files)
    npzs = [np.load(f) for f in files]
    data_list   = [npz['data'] for npz in npzs]
    labels_list = [npz['labels'] for npz in npzs]

    data   = np.concatenate(data_list)
    labels = np.concatenate(labels_list)

    np.savez(outfile, data=data, labels=labels)
