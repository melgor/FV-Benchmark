# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2015-05-28 12:25:23
# @Last Modified 2015-06-15
# @Last Modified time: 2015-06-15 16:27:32
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


descriptors_path = os.path.join(base_path, "descriptors")
databases_path = os.path.join(base_path, "databases")
models_path = os.path.join(base_path, "models")

''' SECTION: DATABASE, name of files to read'''
name_read_labels      = "labels.txt"
name_read_train       = "train.txt"
name_read_val         = "val.txt"
name_read_train_ver_1 = "train_ver_1.txt"
name_read_train_ver_2 = "train_ver_2.txt"
name_read_val_ver_1   = "val_ver_1.txt"
name_read_val_ver_2   = "val_ver_2.txt"

''' SECTION: SAVED DATABASE, name of files for dataset and features to save'''
name_data_color    = 'data.npy'
name_data_gray     = 'data_gray.npy'
name_data_filename = 'data_filename.txt'
name_labels        = 'labels.npy'
name_features      = 'features.npy'
name_features_mirror = 'features_mirrored.npy'
name_label_train   = 'labels_train.npy'
name_label_val     = 'labels_val.npy'
name_ver_train     = 'labels_ver_train.npy'
name_ver_val       = 'labels_ver_val.npy'
in_one_file        = 1000

''' SECTION: FEATURES '''
LBP, ULBP, DEEP = "lbp", "ulbp","deep"
descriptor_types = [LBP, ULBP,DEEP]

#Caffe setting
caffe_models_folder      =  '/home/blcv/CODE/Caffe_Utils/Predict_Caffe/CASIA_Ver_conv52/nets/'
caffe_proto_path_extract = caffe_models_folder + 'casia_extract.prototxt'
caffe_proto_path_deploy  = caffe_models_folder  + 'casia_deploy.prototxt'
caffe_net_path           = caffe_models_folder + '_iter_400001.caffemodel'
caffe_mean_image         = caffe_models_folder + 'mean_file.npy'
caffe_image_size         = (100, 100)
caffe_gpu_id             = 1

#LBP settings
size_image = (152,82)
normalize = False

''' SECTION: LFW, setting'''
lfw_pairs_file       = "pairs.txt"
lfw_list_images_file = "list_images_lfw.txt"
lfw_data_path        = "lfw_data.npy"
lfw_pairs_path       = "pair_labels.npy"
lfw_name             = "LFW"

#BLUFR on LFW settings
blufr_lfw_image_file    = "image_list.txt"
blufr_lfw_labels        = "labels.txt"
blufr_lfw_test_set      = "test_set.txt"
blufr_lfw_gallery       = "gallery_set.txt"
blufr_lfw_probe         = "probe_set.txt"
blufr_lfw_path_to_lfw   = "path_image_lfw.txt"
blufr_lfw_data_path     = "blufr_lfw_data.npy"
blufr_lfw_label_path    = "blufr_lfw_label.npy"
blufr_lfw_test_set_path = "blufr_lfw_test_set.npy"
blufr_lfw_gallery_path  = "blufr_lfw_gallery.npy"
blufr_lfw_probe_path    = "blufr_lfw_probe.npy"
blufr_lfw_test_ver_path = "Ver/blufr_lfw_test_ver_{0}_{1}.npy"
blufr_lfw_test_ver_file = "Ver/blufr_lfw_test_ver.txt"
blufr_lfw_ver_part      = 0.1