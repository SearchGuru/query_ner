from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle
import os
import string

import deepctxt_util
from deepctxt_util import DCTokenizer

batch_size = 1
nb_word = 4
nb_tag = 2
maxlen = 5


x_train = [[1,2],[1,3]] #two sequences, one is [1,2] and another is [1,3]
y_train = [[[0,1],[1,0]],[[0,1],[1,0]]] #the output should be 3D and one-hot for softmax output with categorical_crossentropy
x_test = [[1,2],[1,3]]
y_test = [[[0,1],[1,0]],[[0,1],[1,0]]]

X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
X_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#Y_train = np.asarray(y_train, dtype='int32')
Y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
#Y_test = np.asarray(y_test, dtype='int32')
Y_test = sequence.pad_sequences(y_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(nb_word, 128))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributedDense(nb_tag))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=4, show_accuracy=True)
res = model.predict_classes(X_test)
print('res',res)

