#!/usr/bin/env python
# FV-Benchmark

# Make sure that caffe is on the python patht this file is expected to be in {caffe_root}/examples
caffe_root = '/home/blcv/LIB/caffe_master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import os
import argparse
import shutil
import cv2
import numpy as np

execfile("/media/blcv/drive_2TB/CODE/FV-Benchmark/lib/dataset/fix_imports.py")
import config
from utils import *
from transform_data import loadClassVerData


def loadImage(path, color):
    image_ = caffe.io.load_image(path,color=color)
    if image_ is None or image_.shape[0] == 100:
      print "Error with image", path
      return None
    #image_ =  caffe.io.resize_image(image_, 
    image_ =  cv2.resize(image_, (100, 100))
    return image_

def loadImagesList(list_image, color):
    data = []
    labels = []
    for image in list_image:
      path, number = image
      image = loadImage( path, color)
      if image == None:
        continue
      data.append(image)
      labels.append(int(number))
        
    return np.asarray(data).astype(np.float16), labels
  
def loadImagesDatabase(directory, color, path_to_save):
    #Load images to numpy array
    names_file = os.path.join(directory, config.name_read_labels)
    # Save the result
    if args.color:
      name_data = config.name_data_color
    else:
      name_data = config.name_data_gray
      
    labels_all = []
    filenames = []
    with open(names_file, 'r') as f:
      lines = [ line.strip().split(' ') for line in f.readlines()]
      for i in range(0, len(lines), config.in_one_file):
        print "read: ", config.in_one_file
        if (i+config.in_one_file) < len(lines):
          curr_list = lines[i:i + config.in_one_file] 
        else:
          curr_list = lines[i:] 
        filename = os.path.join(path_to_save, str(i) + '_' + name_data)
        data, labels = loadImagesList(curr_list, color )
        
        print "data readed, now saving", filename
        
        filenames.append(filename)
        np.save(filename, data)
        labels_all.extend(labels)
        
    print "Save Data"    
    filename = os.path.join(path_to_save,config.name_labels)
    np.save(filename, labels_all)
    filename_data = os.path.join(path_to_save,config.name_data_filename)
    with open(filename_data, 'wb') as f:
      f.writelines( "%s\n" % item for item in filenames )
    print "Data Saved"


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Format Any datasets with all preprocessing done and files with absolute path to example")
    parser.add_argument("--color", action="store_true",help="gray or color")
    parser.add_argument("--ver", action="store_true",help="compute label idx and verification, ones done for each datset")
    parser.add_argument("directory", help="ImageSet directory, need to be there labels and: train, val and verification files ")
    parser.add_argument("namedatabase", help="name to give to the dataset")
    args = parser.parse_args()

    path_to_save = os.path.join(config.databases_path,args.namedatabase)
    create_dir(path_to_save)
    # Load the dataset and format it properly
    print "Read Images"
    #loadImagesDatabase(args.directory, args.color,path_to_save )

    if args.ver:
      print "Start Verification Task"
      # Load training and validation set for classification and  verification
      class_train, class_val, ver_train, ver_val = loadClassVerData(args.directory)
      #Save Verification Data
      filename = os.path.join(path_to_save,config.name_label_train)
      np.save(filename, class_train)
      filename = os.path.join(path_to_save,config.name_label_val)
      np.save(filename, class_val)
      filename = os.path.join(path_to_save,config.name_ver_train)
      np.save(filename, ver_train)
      filename = os.path.join(path_to_save,config.name_ver_val)
      np.save(filename, ver_val)
      print "End of preparation verification Task"

    print "Data saved to %s" % path_to_save


