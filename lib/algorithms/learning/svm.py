# -*- coding: utf-8 -*-
# @Author: blcv
# @Date:   2015-06-09 11:41:21
# @Last Modified 2015-06-18
# @Last Modified time: 2015-06-18 13:51:47
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from ..distance import *
from ..verification import *
from ..scaler import *
'''
Learn SVM based on distance of features.
1. Give features, type of distance and verification data (in format 'ID ID 0/1')
2. Transform features to new representation
3. Learn SVM using new representation and labels 0/1
4. Predict value
'''

class SVM(object):
  """docstring for SVM"""
  def __init__(self, ground_truth):
    self.max_num = 3.0
    self.ground_truth = ground_truth
    self.scale = Scaler((0,1))

  #scale all features to be between 0-1        
  def scaleDataFit(self, data):
      self.scale.fit(data)
      return self.scale.transform(data)

  def fit(self, X, Y, distance = chi2Distance):
    X = self.scaleDataFit(X)
    self.distance = distance
    #compute distances
    match, mis = threadComputeMatrix(X, Y, distance)
    #prepare labels
    labels = [1 for l in match ]
    lab  = [0 for l in mis ]
    #merge matches and mismatches
    match = np.vstack((match,mis))
    del mis
    labels.extend(lab)
    match  = np.asarray(match)
    labels = np.asarray(labels)
    #learn
    self.clf = SGDClassifier(loss="hinge", penalty="l2", n_jobs=8, shuffle = True)
    self.clf.fit(match, labels)
    #self.clf = SVC()
    #self.clf.fit(match, labels)
    # print "ACC", self.clf.score( match ,labels)
    # data_match = self.clf.decision_function( np.asarray(match) )
    # data_mis   = self.clf.decision_function( np.asarray(mis  ) )
    # return (data_match, data_mis)
    
  #transform data for evaulation purpose
  def transform(self, X, Y):
    X =  self.scale.transform(X)
    match, mis = threadComputeMatrix(X, Y, self.distance)
    #match, mis = computeDistanceMatrix(X, Y, self.distance)
    data_match = self.clf.decision_function( np.asarray(match) )
    data_mis   = self.clf.decision_function( np.asarray(mis  ) )
    return (data_match, data_mis)
  
 
  def predict(self, X1, X2):
    stack  = np.vstack((X1,X2))
    stack =  self.scale.transform(stack)
    distance = self.distance(stack[0,:],stack[1,:])
    return self.clf.decision_function( np.asarray(distance) )
    
