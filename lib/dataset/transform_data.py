# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2015-05-28 13:07:03
# @Last Modified 2015-05-28
# @Last Modified time: 2015-05-28 15:20:50

import os
execfile("/media/blcv/drive_2TB/CODE/FV-Benchmark/lib/dataset/fix_imports.py")
import config
import numpy as np

def readTxtFile(path):
  with open(path, 'r') as f:
     data  = [line.strip() for line in f.readlines()]
  return data

def createVerification(labels, file_1, file_2):
  #get idx from label files
  label_idx_file_1 = [ labels.index(item)  for item in file_1]
  label_idx_file_2 = [ labels.index(item)  for item in file_2]

  #crete verification by taking by one example from each list and compering their Label
  labels_ver = list()
  for file1,file2 in zip(file_1, file_2):
      lab_1 = file1.split(' ')[1]
      lab_2 = file2.split(' ')[1]
      labels_ver.append(int(lab_1==lab_2))
  
  #merge all data to one numpy array
  mergerd = np.vstack((np.asarray(label_idx_file_1), np.asarray(label_idx_file_2),np.asarray(labels_ver))).T
  #divide array to two subset: matches and mismatches
  list_matches = []
  list_mismatches = []
  for idx,i in enumerate(mergerd):
    if i[2] == 1:
      list_matches.append(idx)
    else:
      list_mismatches.append(idx)

  
  matches = mergerd[list_matches]
  mismatches = mergerd[list_mismatches]
  
  return (matches, mismatches)

def loadClassVerData(directory):
  names_labels = os.path.join(directory, config.name_read_labels)
  names_train  = os.path.join(directory, config.name_read_train)
  names_val    = os.path.join(directory, config.name_read_val)
  labels       = readTxtFile(names_labels)
  labels_train = readTxtFile(names_train)
  labels_val   = readTxtFile(names_val)

  print "Create IDX to labels"
  #Create idx of train anc val data, wher idx point to row in labels
  label_idx_train = [ labels.index(item)  for item in labels_train]
  label_idx_val = [ labels.index(item)  for item in labels_val]
  print "End of creating IDX"
  
  #Create Verification data
  names_train_ver_1 = os.path.join(directory, config.name_read_train_ver_1)
  names_train_ver_2 = os.path.join(directory, config.name_read_train_ver_2)
  names_val_ver_1   = os.path.join(directory, config.name_read_val_ver_1)
  names_val_ver_2   = os.path.join(directory, config.name_read_val_ver_2)

  train_ver_1 = readTxtFile(names_train_ver_1)
  train_ver_2 = readTxtFile(names_train_ver_2)
  val_ver_1   = readTxtFile(names_val_ver_1)
  val_ver_2   = readTxtFile(names_val_ver_2)
  print "Create IDX to Verification"
  train_ver   = createVerification(labels, train_ver_1, train_ver_2)
  val_ver     = createVerification(labels,   val_ver_1,   val_ver_2)
  print "End of creating IDX to Verification"
  
  return np.asarray(label_idx_train), np.asarray(label_idx_val), train_ver, val_ver
