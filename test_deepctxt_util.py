import os
import string

import deepctxt_util
from deepctxt_util import DCTokenizer


# initialize class name to Id mapping table
className2Id = dict()
className2Id['O'] = 0
className2Id['B_ORGANIZATION'] = 1
className2Id['I_ORGANIZATION'] = 2
className2Id['B_LOCATION'] = 3
className2Id['I_LOCATION'] = 4
className2Id['B_PERSON'] = 5
className2Id['I_PERSON'] = 6

print('Loading data... (Train)')
(X1, y_train) = deepctxt_util.load_sequence_data_x_y('./data/test.tsv', className2Id)
print('Done')
