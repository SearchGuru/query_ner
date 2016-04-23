from __future__ import print_function
import numpy as np
np.random.seed(1337)  


import deepctxt_util
from deepctxt_util import DCTokenizer

maxlen = 25 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 3

print('Loading data... (Test)')
(X2, y_test) = deepctxt_util.load_raw_data_x_y(path='./raw_data/person_birthday_deep_learning_eval_rawquery.tsv')
print('Done')

