# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2014-11-27 11:54:32
# @Last Modified 2015-02-27
# @Last Modified time: 2015-02-27 22:43:31
import numpy as np
import json
import cPickle
import os
import gzip
import sys
from collections import namedtuple
fv_bench = '/home/blcv/drive_2TB/CODE/FV-Benchmark/lib/' #import package  from FV-Benchmark
sys.path.insert(0, fv_bench)
from test_multi import *


def cosineDistance(x, y):
    return np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
  
def chi2Distance(x, y):
  feat_div = (x - y)**2
  feat_sum = x + y
  
  print 'feat_div', np.sum(feat_div)
  print 'feat_sum', np.sum(feat_sum)

  return feat_div/feat_sum


def load_cPickle(name_file):
  '''Load file in cPickle format'''
  f = gzip.open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp

'''
sys.argv[1] - Features 
sys.argv[2] - labels  
sys.argv[3] - model
'''
if __name__ == '__main__':
  
  data = np.load(sys.argv[1])
  label_train = np.load(sys.argv[2])
  model       = load_cPickle(sys.argv[3])
  label_ver       = np.load(sys.argv[4])
  features =  data#[label_train,:]
  print "Compute Distance"
  matches, mismatches = model.transform(features, label_ver)
  scores_ = (matches, mismatches)
  print "Compute ACC"
  acc,roc    = computeAccuracyROC(scores_)
  thres   = getBestThreshold(scores_)
  print "Best Thres:" , thres, " ACC: ", acc
  


  
  