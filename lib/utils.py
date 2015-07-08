# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2015-05-28 12:25:23
# @Last Modified 2015-06-10
# @Last Modified time: 2015-06-10 15:03:02
import numpy as np
import json
import cPickle
import os
import gzip
import json

# Save the result
def create_dir(path):
  if not os.path.exists(path):
      os.makedirs(path)
      
def save_cPickle(data, name_file):
  '''Save file in cPickle format, delete if exist'''
  if os.path.isfile(name_file):
      os.remove(name_file)
  f = gzip.open(name_file,'wb')
  cPickle.dump(data,f,protocol=2)
  f.close()  
  
def load_cPickle(name_file):
  '''Load file in cPickle format'''
  f = gzip.open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp


def load_cPickleNormal(name_file):
  '''Load file in cPickle format'''
  f = open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp

def parse_deep_config(config):
  with open(config,'r') as f:
    config = json.load(f)
    
  config["caffe_proto_path_extract"] = str(config["caffe_models_folder"] + config["caffe_proto_path_extract"])
  config["caffe_proto_path_deploy"]  = str(config["caffe_models_folder"] + config["caffe_proto_path_deploy"])
  config["caffe_net_path"]           = str(config["caffe_models_folder"] + config["caffe_net_path"])
  config["caffe_mean_image"]         = str(config["caffe_models_folder"] + config["caffe_mean_image"])
  sizes                              = config["caffe_image_size"].split(',')
  config["caffe_image_size"]         = (int(sizes[0]), int(sizes[1]))
  config["caffe_gpu_id"]             = int(config["caffe_gpu_id"])
  
  return config