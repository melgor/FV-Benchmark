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


def load_cPickle(name_file):
  '''Load file in cPickle format'''
  f = gzip.open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp


if __name__ == '__main__':
  name = sys.argv[1]
  model = load_cPickle(name)
  min_max = np.asarray(model.scale.list_max_min)
  min_values = min_max[:,0]
  diff_values = min_max[:,2]
  coef_       = model.clf.coef_[0]
  bias        = model.clf.intercept_
  print "min_value:", min_values
  print "bias:",bias
  
  examples.saveSVMModel(coef_ ,bias, min_values, diff_values)
