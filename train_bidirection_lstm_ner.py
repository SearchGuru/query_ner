from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle
import os
import string

import deepctxt_util
from deepctxt_util import DCTokenizer
import encode_category_vector

maxlen = 25 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 30

tokenizer = DCTokenizer()
print('Loading tokenizer')
tokenizer.load('./glove.6B.100d.txt')
#tokenizer.load('./glove.42B.300d.txt')
print('Done')

max_features = tokenizer.n_symbols
vocab_dim = tokenizer.vocab_dim

# initialize class name to Id mapping table
className2Id = dict()
className2Id['O'] = 0
className2Id['B_ORGANIZATION'] = 1
className2Id['I_ORGANIZATION'] = 2
className2Id['B_LOCATION'] = 3
className2Id['I_LOCATION'] = 4
className2Id['B_PERSON'] = 5
className2Id['I_PERSON'] = 6

num_categories = len(className2Id)

print('Loading data... (Train)')
(X1, y_train) = deepctxt_util.load_sequence_data_x_y('./data/train.cleaned.tsv', className2Id)
#(X1, y_train) = deepctxt_util.load_sequence_data_x_y('./data/test.cleaned.tsv', className2Id)
y_train = encode_category_vector.one_hot_category(y_train, num_categories)
print('Done')

print('Loading data... (Test)')
(X3, y_test) = deepctxt_util.load_sequence_data_x_y('./data/test.cleaned.tsv', className2Id)
y_test = encode_category_vector.one_hot_category(y_test, num_categories)
print('Done')

print('Converting data... (Train)')
X_train = tokenizer.texts_to_sequences(X1, maxlen)
print('Done')

print('Converting data... (Test)')
X_test = tokenizer.texts_to_sequences(X3, maxlen)
print('Done')

print(len(X_train), 'y_train sequences')
print(len(X_test), 'y_test sequences')


print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
Y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
Y_test = sequence.pad_sequences(y_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

left = Sequential()
left.add(Embedding(input_dim=max_features, output_dim=vocab_dim, input_length=maxlen, weights=[tokenizer.embedding_weights]))
left.add(LSTM(128, return_sequences=True))

right = Sequential()
right.add(Embedding(input_dim=max_features, output_dim=vocab_dim, input_length=maxlen, weights=[tokenizer.embedding_weights]))
right.add(LSTM(128, return_sequences=True, go_backwards=True))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(Dropout(0.5))
model.add(TimeDistributedDense(num_categories))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

print('Train model...')
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=epoch, 
          validation_split=0.1, show_accuracy=True)

print('Done')

print('Evaluate model...')
score, acc = model.evaluate([X_test, X_test], Y_test, batch_size=100, show_accuracy=True)

print('Test score:', score)
print('Test accuracy:', acc)

json_model_string = model.to_json()
with open("./query_ner_birnn_lstm_glove_"+str(batch_size)+"."+str(epoch)+"b.json", "wb") as f:
    f.write(json_model_string)
model.save_weights("./query_ner_birnn_lstm_glove_" + str(batch_size) + "." + str(epoch) + "b.h5", overwrite=True)


