# -*- coding: utf-8 -*-
from __future__ import absolute_import

import string
import sys
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def one_hot_category(Y, categorySize):
    Y_encode = []
    for y in Y:
        y_encode = np_utils.to_categorical(y, categorySize)
        Y_encode.append(y_encode)
    return Y_encode

