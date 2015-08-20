#!/usr/bin/env python
# FV-Benchmark
import sys
import os
import argparse
import numpy as np
import cv2
execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config
from utils import *
from features.deep.extractor import *
from features.lbp.descriptors import LbpDescriptor


def computeLBPFeatures(data,descriptor_type ):
    #resize all images to proper size
    resize_data = np.asarray([cv2.resize(img, config.size_image)  for img in data])
    pca = None
    lda = None
    descriptor = LbpDescriptor(descriptor_type, pca=pca, lda=lda)
    sample = descriptor.compute(resize_data[0])
    n_samples = resize_data.shape[0]
    n_features = sample.shape[0]
    print "Sample shape", sample.shape
    descriptors = np.empty((n_samples, n_features), dtype=sample.dtype)
    for i in xrange(n_samples):
        descriptors[i] = descriptor.compute(resize_data[i], normalize=config.normalize)
    return descriptors

def computeDescriptors(input_file, descriptor_type, config_net ):
    
    if descriptor_type == config.DEEP:
      print "Extract Features using DEEP: ", config_net["caffe_proto_path_extract"]
      extractor    = Extractor(config_net, config_net["caffe_proto_path_extract"])
      descriptor = []
      for inputf in input_file:
        data = np.load(inputf)
        # data_mirrored = np.fliplr(data)
        desc = extractor.predict_data(data)
        descriptor.append(desc)
        print "Predicted: ", config.in_one_file

    else:
      print "Compute LBP/ULBP features"
      descriptor = []
      for input_file in input_file:
        data = np.load(input_file)
        desc = computeLBPFeatures(data, descriptor_type )
        descriptor.append(desc)
        print "Predicted: ", config.in_one_file
        
    descriptor = np.vstack(descriptor)  
    return descriptor

if __name__ == "__main__":
    
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Computes descriptors")
    parser.add_argument("descriptor_type", choices=config.descriptor_types)
    parser.add_argument("dataset", help="dataset to use (name)")
    parser.add_argument("config", default = None, help="config of network if Deep choosen")
    args = parser.parse_args()
  
    descriptor_type = args.descriptor_type
    if descriptor_type == config.DEEP and args.config == None:
      print "IF deep choosen, need to specify config of net"
      sys.exit()
    
    
    if descriptor_type != config.DEEP:
      config_net  = None
      output_file = os.path.join( config.descriptors_path, args.dataset, "_".join((descriptor_type,config.name_features)))
    else:
      config_net  = parse_deep_config(args.config)
      output_file = os.path.join( config.descriptors_path, args.dataset, "_".join((descriptor_type,config_net["name"] ,config.name_features)))
    
    create_dir(os.path.dirname(output_file))
    # Load data filename
    input_file =  os.path.join(config.databases_path, args.dataset , config.name_data_filename)
    with open(input_file,'r') as f:
      files_path = [line.strip() for line in f]
    
    descriptor = computeDescriptors(files_path, descriptor_type, config_net)
    
    # Save results
    np.save(output_file, descriptor)
    print "Results saved in %s" % output_file
