from shared.Action import Action

import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse, sys, pdb

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.utils import np_utils
import keras.metrics as metrics


class CNNModel():

    def __init__(self, num_epoch=5, batch_size=50, num_filters=24, kernel_size=5, kernel_initializer='glorot_uniform', verbose=True):

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.pool_size = (2, 2)
        self.num_dense = 64

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
        assert type in ['train', 'test']

        if type=='train':
            self._load_train_data(file)
        else:
            self._load_test_data(file)


    def _load_train_data(self, trainfile, verbose=True):
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

        if verbose:
            print(X_train.shape[0], 'train samples')
            print(X_val.shape[0], 'validation samples')
            print('Num. classes: {}'.format(self.num_classes))
            print('Train class counts: {}'.format(np.bincount(y_train)))
            print('Input shape: {}'.format(self.input_shape))


    def _load_test_data(self, testfile, verbose=True):
        (X_test, y_test), (rows, cols, chans) = self._read_data(testfile, split=False)

        self.X['test']    = X_test
        self.y['test']    = y_test
        self.rows         = rows
        self.cols         = cols
        self.chans        = chans
        self.input_shape  = (self.rows, self.cols, self.chans)
        self.num_classes  = np.unique(y_test).size

        self._normalize_data(targets=['test'])

        if verbose:
            print(X_test.shape[0], 'test samples')


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

        # # Convert to range 0.0 - 1.0
        # self.X['train'] = self.X['train'].astype('float32')
        # self.X['val']   = self.X['val'].astype('float32')
        # self.X['test']  = self.X['test'].astype('float32')
        # self.X['train'] /= 255
        # self.X['val']   /= 255
        # self.X['test']  /= 255
        #
        # # convert class vectors to binary class matrices
        # self.Y['train'] = np_utils.to_categorical(self.y['train'], self.num_classes)
        # self.Y['val']   = np_utils.to_categorical(self.y['val'], self.num_classes)
        # self.Y['test']  = np_utils.to_categorical(self.y['test'], self.num_classes)
        #
        # # reshape image for Keras, note that image_dim_ordering set in ~.keras/keras.json
        # K.set_image_data_format('channels_last')
        # self.X['train'] = self.X['train'].reshape((self.X['train'].shape[0], self.rows, self.cols, self.chans))
        # self.X['val'] = self.X['val'].reshape((self.X['val'].shape[0], self.rows, self.cols, self.chans))
        # self.X['test'] = self.X['test'].reshape((self.X['test'].shape[0], self.rows, self.cols, self.chans))


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

        # # Second hidden layer
        # self.add(Dense(num_dense//2))
        # self.add(Activation('relu'))

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
        self.model.fit(self.X['train'], self.Y['train'],
                       batch_size=self.batch_size,
                       epochs=self.num_epoch,
                       verbose=1,
                       class_weight=self.class_weights,
                       validation_data=(self.X['val'], self.Y['val']))


    def predict_proba(self, samples=None):
        if not samples:
            samples = self.X['test']

        # Add logic to handle single sample
        return self.model.predict_proba(samples)


    def predict_class(self, samples=None):
        if not samples:
            samples = self.X['test']

        # Add logic to handle single sample
        return self.model.predict_classes(samples)


    def test(self):
        print('Testing model...')
        y_pred = self.predict_class()
        class_names = Action().names[0:self.num_classes]

        accuracy = accuracy_score(self.y['test'], y_pred)
        print('Accuracy: {}'.format(accuracy))
        print()

        print('Classification report:')
        report = classification_report(self.y['test'], y_pred, target_names=class_names)
        print(report)
        print()

        print('Confusion matrix:')
        confusion = confusion_matrix(self.y['test'], y_pred)
        print(confusion)

        # print('Testing model...')
        # scores = self.model.evaluate(self.X['test'], self.Y['test'], verbose=0)
        #
        # print()
        # for i in range(len(scores)):
        #     print('{}: {}'.format(self.model.metrics_names[i], score[i]))


    def save_model(self, file='model.hdf5'):
        self.model.save(file)


    def load_model(self, file='model.hdf5'):
        self.model = load_model(file)


def get_args():
    parser = argparse.ArgumentParser(description='Build or test convolutional neural network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', nargs=2, metavar=('trainfile', 'testfile'), help='train and test the model')
    group.add_argument('--test', metavar='testfile', help='test the model')
    parser.add_argument('--model', default='model.hdf5', metavar='modelfile', help='model file to save or load')
    parser.add_argument('--epochs', type=int, default=1, metavar='num_epochs', help='Number of epochs to train')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = CNNModel(num_epoch=args.epochs)

    if args.train:
        trainfile, testfile = args.train
        model.load_data(trainfile, type='train')
        model.load_data(testfile, type='test')
        model.build()
        model.print_summary()
        model.fit()
        model.test()

        if args.model:
            model.save_model(args.model)
    else:
        testfile = args.test
        model.load_data(testfile, type='test')
        model.load_model(args.model)
        model.test()
