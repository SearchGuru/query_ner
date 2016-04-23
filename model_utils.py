# -*- coding: utf-8 -*-
from __future__ import absolute_import

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def evaluate_model(Y_pred, Y_true):
    total_item_count = 0
    matched_iterm_count = 0
    mis_match_iterm_count = 0
    if (len(Y_pred) != len(Y_true)):
        print('Input two arrays size does not match')
        exit()

    for i in range(0, len(Y_true)):
        y_pred = Y_pred[i]
        y_true = Y_true[i]
        for j in range(0, len(y_true)):
            k = len(y_pred) - len(y_true) + j
            if (k < 0 or k >= len(y_pred)):
                print('out of range:', k)
                exit()
            if (y_pred[k] == y_true[j]):
                matched_iterm_count += 1
            else:
                mis_match_iterm_count += 1
    total_item_count = matched_iterm_count + mis_match_iterm_count

    acc = 100.0*matched_iterm_count / total_item_count
    return acc
            
