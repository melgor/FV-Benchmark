#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: blcv
# @Date:   2015-06-19 10:29:43
# @Last Modified 2015-06-19
# @Last Modified time: 2015-06-19 11:39:32
import numpy as np
import cPickle
import gzip
import sys
fv_bench = '/media/blcv/drive_2TB/CODE/FV-Benchmark/lib/'
import sys
sys.path.insert(0, fv_bench)
from test_multi import *
import examples

'''
Program transform C++ features to Numpy. Does not extract labels and names (not needed for learning, labels are extracted earlier)
sys.argv[1] is the path to Features extracted by C++ code (not Python, because there is slight difference in diff_values
sys.argv[2] is name of output)
'''

if __name__ == '__main__':
  name_train = sys.argv[1]
  #name_val = sys.argv[2]
  out  = sys.argv[2]
  features_train =  examples.transformFeatures(name_train)
  #features_val =  examples.transformFeatures(name_val)
  #features = np.vstack((features_train,features_val))
  np.save(out, features_train)
