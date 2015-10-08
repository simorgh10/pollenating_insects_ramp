# Put your classifier.py here
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping

import math
from skimage import data
from skimage import transform as tf


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
hidden4_num_units=500, hidden5_num_units=500,
dropout5_p=0.5,
output_num_units=18, output_nonlinearity=nonlinearities.softmax,
update_learning_rate=0.01,
update_momentum=0.9,
max_epochs=200,
on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
batch_iterator_train=FlipBatchIterator(batch_size=256)
)


class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):

        # additional static transformations
        print 'start preprocess'
        tformslist = list()
        tformslist.append(tf.SimilarityTransform(scale=1))
        tformslist.append(tf.SimilarityTransform(scale=1, translation=(16,0)))
        tformslist.append(tf.SimilarityTransform(scale=1, translation=(-16,0)))
        #tformslist.append(tf.SimilarityTransform(scale=1, translation=(0,16)))
        #tformslist.append(tf.SimilarityTransform(scale=1, translation=(0,-16)))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = math.pi/10))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = -math.pi/10))

        X_new = np.zeros((X.shape[0] * 5, X.shape[1], X.shape[2], X.shape[3]))
        print 'X shape ', X.shape[0]
        for i in xrange(X.shape[0]):
            Xbase = X[i,:,:,:]
            if i % 1000 == 0:
                print 'performed first ' + str((i)) + ' transformations.'
            for j in xrange(len(tformslist)):
                X_new[len(tformslist)*i + j, :, :, :] = tf.warp(Xbase, tformslist[j])
        print 'end preprocess'
        X_new = (X_new / 255.)
        X_new = X_new.astype(np.float32)
        X_new = X_new.transpose((0, 3, 1, 2))
        return X_new

    def preprocess_y(self, y):
        y_new = np.zeros((y.shape[0] * 5))
        for i in xrange(y.shape[0]):
            for j in xrange(5):
                y_new[5*i + j] = y[i]
        return y_new.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
