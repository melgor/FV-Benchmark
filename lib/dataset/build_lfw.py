#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: blcv
# @Date:   2015-06-10 16:28:38
# @Last Modified 2015-06-15
# @Last Modified time: 2015-06-15 11:16:46

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

execfile("fix_imports.py")
import config
from utils import *

'''Creating Pair files'''
def getIdx(name,idx,all_people_images):
   if idx < 10:
    name_one ="_".join((name,"000{0}.jpg".format(idx)))
   elif idx < 100:
    name_one ="_".join((name,"00{0}.jpg".format(idx)))
   elif idx < 1000:
    name_one ="_".join((name,"0{0}.jpg".format(idx)))
   
   return all_people_images.index(name_one)
 
#get idx of pairs from good samples
def analyseGood(labels, all_people_images):
   data = labels.strip().split('\t')
   name = data[0]
   idx_one = int(data[1])
   idx_two = int(data[2])
   #generate first name
   idx_one = getIdx(name,idx_one,all_people_images)
   idx_two = getIdx(name,idx_two,all_people_images)
   
   return idx_one,idx_two

#get idx of pairs from bad samples
def analyseBad(labels, all_people_images):
   data = labels.strip().split('\t')
   name_one = data[0]
   idx_one = int(data[1])
   name_two = data[2]
   idx_two = int(data[3])
   #generate first name
   idx_one = getIdx(name_one,idx_one,all_people_images)
   idx_two = getIdx(name_two,idx_two,all_people_images)
   
   return idx_one,idx_two

def createPairFile(directory, path_to_save, name ):
  names_file = os.path.join(directory, config.lfw_pairs_file)
  with open(names_file,'r') as f:
    content = f.readlines()
    
  #get unique id
  splited = content[0].strip().split('\t')
  sets_num = int(splited[0])
  num_in_set =  int(splited[1])

  #read file with all images
  names_file = os.path.join(directory, config.lfw_list_images_file)
  with open(names_file,'r') as f:
    lines = [ line.strip().split(' ')[0] for line in f.readlines()]
    names_people = [line.split('/')[-1]   for line in lines]

  pairs_labels = list()
  for sn in range(sets_num):
    print "Create"
    #first loop by good pairs
    labels_set = content[ 1 + 2 * sn * num_in_set : 1 + 2 * ( sn + 1 ) * num_in_set ]
    print 1+2*sn*num_in_set, 1 + 2*(sn+1)*num_in_set
    good_labels = labels_set[:num_in_set]
    bad_labels = labels_set[num_in_set:]
    print len(labels_set), len(good_labels), len(bad_labels)
    for g in good_labels:
      idx_one,idx_two = analyseGood(g,names_people)
      pairs_labels.append((idx_one,idx_two, 1))
    for b in bad_labels:
      idx_one,idx_two = analyseBad(b,names_people)
      pairs_labels.append((idx_one,idx_two, 0))
      
  
  #divide array to two subset: matches and mismatches
  pairs_labels = np.asarray(pairs_labels)
  list_matches = []
  list_mismatches = []
  for idx,i in enumerate(pairs_labels):
    if i[2] == 1:
      list_matches.append(idx)
    else:
      list_mismatches.append(idx)

  
  matches = pairs_labels[list_matches]
  mismatches = pairs_labels[list_mismatches]
  print "all pairs: ", len(pairs_labels)
  names_file = os.path.join(path_to_save, name + "_" + config.lfw_pairs_path) 
  np.save(names_file, (matches, mismatches) )

'''Loading DataBase'''
def loadImage(path, color):
    image_ = caffe.io.load_image(path,color=color)
    # if image_ is None or image_.shape[0] == 100:
    #   print "Error with image", path
    #   return None
    image_ =  caffe.io.resize_image(image_, (100, 100))
    #image_ = cv2.resize(image_, (55, 47)) 
    return image_

def loadImagesList(list_image, color):
    data = []
    labels = []
    for image in list_image:
        image = loadImage( image, color)
        data.append(image)
        
    return np.asarray(data).astype(np.float16)
  
def loadImagesDatabase(directory, color, path_to_save, name):
    #Load images to numpy array
    names_file = os.path.join(directory, config.lfw_list_images_file)
    filenames = []
    with open(names_file, 'r') as f:
      lines = [ line.strip().split(' ')[0] for line in f.readlines()]
      data = loadImagesList(lines, color)
      filename = os.path.join(path_to_save, name + "_" +  config.lfw_data_path)
      np.save(filename, data)
      filenames.append(filename)
      filename_data = os.path.join(path_to_save, config.name_data_filename)
      with open(filename_data, 'wb') as f:
        f.writelines( "%s\n" % item for item in filenames )
      print "Data Saved"

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Create LFW dataset")
    parser.add_argument("--color", action="store_true",help="gray or color")
    parser.add_argument("directory", help="LFW directory with list_images and pairs")
    parser.add_argument("namelfw", help="name of dataset, based on size of image (for CASIA LFW_100, for DEEPID2 LFW)")
    parser.add_argument("name", help="name of dataset, based on alignment method")
    args         = parser.parse_args()
    namedatabase = args.namelfw
    path_to_save = os.path.join(config.databases_path, namedatabase)
    create_dir(path_to_save)
    # Load the dataset and format it properly
    print "Read Images"
    loadImagesDatabase(args.directory, args.color, path_to_save, args.name )
    createPairFile(args.directory, path_to_save, args.name )

    print "Data saved to %s" % path_to_save