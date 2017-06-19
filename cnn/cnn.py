#!/usr/bin/env python
import numpy as np
np.random.seed(1337)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse, sys, os, pdb

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.utils import np_utils
import keras.metrics as metrics

sys.path.append(os.path.abspath('..'))
from shared.action import Action


class CNNModel():

    def __init__(self, num_epoch=5, num_hidden=1, batch_size=50, num_filters=24, kernel_size=5, kernel_initializer='glorot_uniform', verbose=True):

        self.num_epoch          = num_epoch
        self.num_hidden         = num_hidden
        self.batch_size         = batch_size
        self.num_filters        = num_filters
        self.kernel_size        = kernel_size
        self.kernel_initializer = kernel_initializer
        self.pool_size          = (2, 2)
        self.num_dense          = 64
        self.verbose            = verbose

        # Other instance variables, for the record...
        self.X             = dict()
        self.y             = dict()
        self.Y             = dict()
        self.rows          = 0
        self.cols          = 0
        self.chans         = 0
        self.num_classes   = 0
        self.class_weights = None
        self.input_shape   = None
        self.model         = None # Has-a model, so not to conflict with state

        K.set_image_data_format('channels_last')

        if verbose:
            print('Epochs: {}'.format(num_epoch))
            print('Batch size: {}'.format(batch_size))
            print('Num filters: {}'.format(num_filters))
            print('Kernel size: {}'.format(kernel_size))
            print('kernel_initializer: {}'.format(kernel_initializer))
            print('Image data format: {}'.format(K.image_data_format()))


    def load_data(self, file, type):
        assert type in ['train', 'test', 'train_and_validate', 'validate']

        if type=='train':
            self._load_train_data(file)
        elif type=='test':
            self._load_test_data(file)
        elif type=='train_and_validate':
            self._load_train_and_validate_data(file)
        elif type=='validate':
            self._load_validate_data(file)
        else:
            assert False

    def _load_train_data(self, trainfile):
        (X_train, y_train), (rows, cols, chans) = self._read_data(trainfile, split=False)

        self.X['train']   = X_train
        self.y['train']   = y_train
        self.rows         = rows
        self.cols         = cols
        self.chans        = chans
        self.input_shape  = (self.rows, self.cols, self.chans)


        self.num_classes   = np.unique(y_train).size
        self.class_weights = dict(enumerate(y_train.size / (self.num_classes * np.bincount(y_train))))

        self._normalize_data(targets=['train'])

        if self.verbose:
            print('{} train samples'.format(X_train.shape[0]))
            print('Num. classes: {}'.format(self.num_classes))
            print('Train class counts: {}'.format(np.bincount(y_train)))
            print('Input shape: {}'.format(self.input_shape))

    def _load_validate_data(self, trainfile):
        (X_val, y_val), (rows, cols, chans) = self._read_data(trainfile, split=False)

        self.X['val']     = X_val
        self.y['val']     = y_val
        self.rows         = rows
        self.cols         = cols
        self.chans        = chans
        self.input_shape  = (self.rows, self.cols, self.chans)

        if self.num_classes > 0:
            assert self.num_classes == np.unique(y_val).size

        self._normalize_data(targets=['val'])

        if self.verbose:
            print('{} validation samples'.format(X_val.shape[0]))


    def _load_train_and_validate_data(self, trainfile):
        (X_train, X_val, y_train, y_val), (rows, cols, chans) = self._read_data(trainfile, split=True)

        self.X['train']   = X_train
        self.X['val']     = X_val
        self.y['train']   = y_train
        self.y['val']     = y_val
        self.rows         = rows
        self.cols         = cols
        self.chans        = chans
        self.input_shape  = (self.rows, self.cols, self.chans)


        self.num_classes   = np.unique(y_train).size
        self.class_weights = dict(enumerate(y_train.size / (self.num_classes * np.bincount(y_train))))

        self._normalize_data(targets=['train', 'val'])

        if self.verbose:
            print('{} train samples'.format(X_train.shape[0]))
            print('{} validation samples'.format(X_val.shape[0]))
            print('Num. classes: {}'.format(self.num_classes))
            print('Train class counts: {}'.format(np.bincount(y_train)))
            print('Input shape: {}'.format(self.input_shape))


    def _load_test_data(self, testfile):
        (X_test, y_test), (rows, cols, chans) = self._read_data(testfile, split=False)

        self.X['test']    = X_test
        self.y['test']    = y_test
        self.rows         = rows
        self.cols         = cols
        self.chans        = chans
        self.input_shape  = (self.rows, self.cols, self.chans)
        self.num_classes  = np.unique(y_test).size

        self._normalize_data(targets=['test'])

        if self.verbose:
            print('{} test samples'.format(X_test.shape[0]))


    def _read_data(self, file, split=False):
        npz = np.load(file)
        data = npz['data']
        labels = npz['labels']
        n = data.shape[0]
        w = 214
        h = 120
        size = data[0].size
        chans = size//w//h

        assert chans==1 or chans==3, "Unexpected buffer size (%d x %d x %d)" % (w, h, chans)

        assert size == w*h*chans, "Unexpected buffer size (%d)" % size

        X = data
        y = labels

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            data_return = (X_train, X_test, y_train, y_test)
        else:
            data_return = (X, y)

        return data_return, (h, w, chans)


    def _normalize_data(self, targets):

        for target in targets:
            self.X[target] = self.X[target].astype('float32')
            self.X[target] /= 255
            self.Y[target] = np_utils.to_categorical(self.y[target], self.num_classes)
            self.X[target] = self.X[target].reshape((self.X[target].shape[0], self.rows, self.cols, self.chans))


    def _normalize_sample(self, sample):
        data = sample.astype('float32')
        data /= 255
        return data


    def build(self):
        self.model = Sequential()

        self.model.add(Conv2D(self.num_filters, self.kernel_size,
                         input_shape=self.input_shape,
                         kernel_initializer=self.kernel_initializer)) #1st conv. layer
        self.model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

        self.model.add(Conv2D(self.num_filters, self.kernel_size,
            kernel_initializer=self.kernel_initializer)) #2nd conv. layer
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(self.num_filters, self.kernel_size,
            kernel_initializer=self.kernel_initializer)) #3rd conv. layer
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=self.pool_size)) # decreases size, helps prevent overfitting
        self.model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

        self.model.add(Flatten()) # necessary to flatten before going into conventional dense layer

        # now start a typical neural network
        self.model.add(Dense(self.num_dense,
            kernel_initializer=self.kernel_initializer))

        self.model.add(Activation('relu'))

        if self.num_hidden > 1:
            self.model.add(Dense(self.num_dense//2))
            self.model.add(Activation('relu'))

        self.model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

        self.model.add(Dense(self.num_classes,
            kernel_initializer=self.kernel_initializer)) # Final nodes
        self.model.add(Activation('softmax')) # keep softmax at end to pick between classes

        # many optimizers available
        # see https://keras.io/optimizers/#usage-of-optimizers
        # suggest you keep loss at 'categorical_crossentropy' for this multiclass problem,
        # and metrics at 'accuracy'
        # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
        # how are we going to solve and evaluate it:
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])


    def print_summary(self):
        self.model.summary()


    def fit(self):
        if verbose:
            verbosity = 1
        else:
            verbosity = 0

        self.model.fit(self.X['train'], self.Y['train'],
                       batch_size=self.batch_size,
                       epochs=self.num_epoch,
                       verbose=verbosity,
                       class_weight=self.class_weights,
                       validation_data=(self.X['val'], self.Y['val']))


    def predict_proba(self, samples=None):
        if samples is None:
            samples = self.X['test']
        return self.model.predict_proba(samples)

    def predict_sample_proba(self, sample):
        data = self._normalize_sample(sample)
        return self.model.predict_proba(data[np.newaxis,:], verbose=0)[0]

    def predict_classes(self, samples=None):
        if samples is None:
            samples = self.X['test']
        return self.model.predict_classes(samples)

    def predict_sample_class(self, sample):
        data = self._normalize_sample(sample)
        return self.model.predict_classes(data[np.newaxis,:], verbose=0)[0]

    def test(self):
        print('Testing model...')
        y_pred = self.predict_classes()
        class_names = Action.names[0:self.num_classes]

        accuracy = accuracy_score(self.y['test'], y_pred)
        print('Accuracy: {}'.format(accuracy))
        print('')

        print('Classification report:')
        report = classification_report(self.y['test'], y_pred, target_names=class_names)
        print(report)
        print('')

        print('Confusion matrix:')
        confusion = confusion_matrix(self.y['test'], y_pred)
        print(confusion)

        # print('Testing model...')
        # scores = self.model.evaluate(self.X['test'], self.Y['test'], verbose=0)
        #
        # print('')
        # for i in range(len(scores)):
        #     print('{}: {}'.format(self.model.metrics_names[i], score[i]))


    def save_model(self, file='model.hdf5'):
        self.model.save(file)


    def load_model(self, file='model.hdf5'):
        self.model = load_model(file)


def get_args():
    parser = argparse.ArgumentParser(description='Build or test convolutional neural network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', metavar='<trainfile>', help='train the model')
    group.add_argument('--test', metavar='<testfile>', help='test the model')
    group.add_argument('--train_and_test', nargs=2, metavar=('<trainfile>', '<testfile>'), help='train and test the model')
    group.add_argument('--resume', metavar='<trainfile>', help='reload model and resume training')
    group.add_argument('--resume_and_test', nargs=2, metavar=('<trainfile>', '<testfile>'), help='reload model, resume training, then test')
    parser.add_argument('--validate', metavar='<valfile>', help='data for validation')
    parser.add_argument('--model', default='model.hdf5', metavar='<modelfile>', help='model file to save or load')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=50, metavar='B', help='Mini-batch size')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verbose', default=True, action='store_true', help='make chatty')
    group.add_argument('--quiet', default=False, action='store_true', help='make quiet')
    parser.add_argument('--hidden', type=int, default=1, metavar='H', help='number of hidden layers to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    verbose = not args.quiet

    model = CNNModel(num_epoch=args.epochs,
                     num_hidden=args.hidden,
                     verbose=verbose,
                     batch_size=args.batch_size)

    if args.train:
        trainfile = args.train
        if args.validate:
            valfile = args.validate
            model.load_data(trainfile, type='train')
            model.load_data(valfile, type='validate')
        else:
            model.load_data(trainfile, type='train_and_validate')
        model.build()
        if verbose:
            model.print_summary()
        model.fit()
        model.save_model(args.model)
    elif args.train_and_test:
        trainfile, testfile = args.train_and_test
        if args.validate:
            valfile = args.validate
            model.load_data(trainfile, type='train')
            model.load_data(valfile, type='validate')
        else:
            model.load_data(trainfile, type='train_and_validate')
        model.load_data(testfile, type='test')
        model.build()
        if verbose:
            model.print_summary()
        model.fit()
        model.test()
        model.save_model(args.model)
    elif args.test:
        testfile = args.test
        model.load_data(testfile, type='test')
        model.load_model(args.model)
        model.test()
    elif args.resume:
        trainfile = args.resume
        if args.validate:
            valfile = args.validate
            model.load_data(trainfile, type='train')
            model.load_data(valfile, type='validate')
        else:
            model.load_data(trainfile, type='train_and_validate')
        model.load_model(args.model)
        if verbose:
            model.print_summary()
        model.fit()
        model.save_model(args.model)
    elif args.resume_and_test:
        trainfile, testfile = args.resume_and_test
        if args.validate:
            valfile = args.validate
            model.load_data(trainfile, type='train')
            model.load_data(valfile, type='validate')
        else:
            model.load_data(trainfile, type='train_and_validate')
        model.load_data(testfile, type='test')
        model.load_model(args.model)
        if verbose:
            model.print_summary()
        model.fit()
        model.test()
        model.save_model(args.model)
    else:
        assert False, 'Unknown command'
