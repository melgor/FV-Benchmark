import numpy as np
import os
execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config
from utils import *

class DataSet(object):
  """docstring for DataSet"""
  def __init__(self, descriptor_type, config_net = None):
    self.name            = config_net['dataset']
    self.config_net      = config_net
    self.descriptor_type = descriptor_type
    # if config_net != None:
    self.path_feat = os.path.join(config.descriptors_path, self.name, "_".join((descriptor_type, config_net['name'], config.name_features)))
    self.path_feat_mirrored = os.path.join(config.descriptors_path, self.name, "_".join((descriptor_type, config_net['name'], config.name_features_mirror)))
    # else:
    #   self.path_feat = os.path.join(config.descriptors_path, self.name, "_".join((descriptor_type,config.name_features)))
      
    path_label = os.path.join(config.databases_path, self.name, config.name_labels)
    self.labels = np.load(path_label)
  

  def loadPathRawData(self):
    path_data = os.path.join(config.databases_path, self.name, config.name_data_filename)
    with open(path_data,'r') as f:
      files_path = [line.strip() for line in f]
    return files_path

  def loadDataClassification(self, type_data):
    if type_data == "train":
      path_labels = os.path.join(config.databases_path, self.name, config.name_label_train)
    elif type_data == "val":
      path_labels = os.path.join(config.databases_path, self.name, config.name_label_val)

    labels   = np.load(path_labels)
    return labels

  def loadDataVerification(self, type_data):
    if type_data == "train":
      path_labels = os.path.join(config.databases_path, self.name, config.name_ver_train)
    elif type_data == "val":
      path_labels = os.path.join(config.databases_path, self.name, config.name_ver_val)

    labels   = np.load(path_labels)
    return  labels
  
  def loadFeatures(self):
    self.features = np.load(self.path_feat)
    return self.features 
  
  def loadFeaturesMirrored(self):
    self.features = np.load(self.path_feat_mirrored)
    return self.features

  def loadLFW(self):
    '''Load LFW database using same net like in config file. Return all data and ground truth which point pairs'''
    path_feat   =  os.path.join(config.descriptors_path, self.config_net['lfw_folder'], "_".join((self.descriptor_type, self.config_net['name'], config.name_features)))
    path_labels =  os.path.join(config.databases_path, self.config_net['lfw_folder'], self.config_net['lfw_name'] + '_' + config.lfw_pairs_path)
    features    =  np.load(path_feat)
    pairs       =  np.load(path_labels)
    return  features.astype(np.float32), pairs
  
  def loadLFWMirrored(self):
    path_feat   =  os.path.join(config.descriptors_path, self.config_net['lfw_folder'], "_".join((self.descriptor_type, self.config_net['name'], config.name_features_mirror)))
    features    =  np.load(path_feat)
    return  features
  
  def loadBLUFR(self):
    '''Load BLUFR data: features and verification data'''
    path_feat   =  os.path.join(config.descriptors_path, self.config_net['blufr_folder'], "_".join((self.descriptor_type, self.config_net['name'], config.name_features)))
    path_pairs  =  os.path.join(config.databases_path, self.config_net['blufr_folder'], config.blufr_lfw_test_ver_file)
    features    =  np.load(path_feat)
    with open(path_pairs,'r') as f:
      pair_files = [ line.strip() for line in  f.readlines()]

    return features, pair_files
    


