from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle

import deepctxt_util
from deepctxt_util import DCTokenizer
import encode_category_vector
import model_utils

maxlen = 25 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 50

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


print('Loading data... (Test)')
(x_test, y_test_class) = deepctxt_util.load_sequence_raw_data_x_y('./data/test.cleaned.tsv')

y_test = []
for class_name_array in y_test_class:
    y = []
    for class_name in class_name_array:
        class_id = className2Id[class_name]
        y.append(class_id)
    y_test.append(y)  

y_test = encode_category_vector.one_hot_category(y_test, num_categories)
print('Done')

print('Converting data... (Test)')
x_test = tokenizer.texts_to_sequences(x_test, maxlen)
print('Done')
print(len(x_test), 'y_test sequences')

#print("Pad sequences (samples x time)")
X_test = sequence.pad_sequences(x_test, maxlen=maxlen)
Y_test = sequence.pad_sequences(y_test, maxlen=maxlen)
#print('X_test shape:', X_test.shape)

print('Load model...')

file = open('./query_ner_birnn_lstm_glove_100.15b.json', 'rb')
model_string = file.read()
file.close()
model = model_from_json(model_string)
model.load_weights('./query_ner_birnn_lstm_glove_100.15b.h5')

print('Evaluate LSTM model...')
score, acc = model.evaluate([X_test, X_test], Y_test, batch_size=100, show_accuracy=True)
print('score:', score)
print('accuracy:', acc)

print('Predict LSTM model...')
Y_pred = model.predict_classes([X_test, X_test], batch_size=100, verbose=1)
# map id to class name
Y_pred_class = []

for class_ids in Y_pred:
    y_pred_class = []
    for y_id in class_ids:
        if (0 == y_id):
            y_pred_class.append('O')
        elif (1 == y_id):
            y_pred_class.append('B_ORGANIZATION')
        elif (2 == y_id):
            y_pred_class.append('I_ORGANIZATION')
        elif (3 == y_id):
            y_pred_class.append('B_LOCATION')
        elif (4 == y_id):
            y_pred_class.append('I_LOCATION')
        elif (5 == y_id):
            y_pred_class.append('B_PERSON')
        elif (6 == y_id):
            y_pred_class.append('I_PERSON')
        else:
            print('Unknown class id:', y_id)
    Y_pred_class.append(y_pred_class)

#acc = model_utils.evaluate_model(Y_pred_class, y_test_class)
#print('accuracy w/o sequence pad:', acc)

#out_file = open('predict_out_ner.tsv', 'wb')
#for i in range(0, len(Y_pred)):
 #   s = X2[i] + '\t' + str(y_test[i]) + '\t' + str(Y_prob[i][0]) + '\t' + str(Y_prob[i][1]) + '\n'
  #  out_file.write(s)
#out_file.close()

total_item_count = 0
matched_iterm_count = 0
mis_match_iterm_count = 0
if (len(Y_pred_class) != len(y_test_class)):
    print('Input two arrays size does not match')
    exit()

for i in range(0, len(y_test_class)):
    y_pred = Y_pred_class[i]
    y_true = y_test_class[i]
    has_mis_match = 0
    for j in range(0, len(y_true)):
        k = len(y_pred) - len(y_true) + j
        if (k < 0 or k >= len(y_pred)):
            #print('out of range:', k)
            continue
        if (y_pred[k].lstrip('I_').lstrip('B_') == y_true[j].lstrip('I_').lstrip('B_')):
            matched_iterm_count += 1
        else:
            mis_match_iterm_count += 1
            has_mis_match = 1
    if (has_mis_match == 1):
        print('mis match:', i)
total_item_count = matched_iterm_count + mis_match_iterm_count
acc = 100.0*matched_iterm_count/total_item_count

print('Accuracy w/o padding:', acc);

