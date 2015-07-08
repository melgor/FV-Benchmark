import os, sys
import cv2
import startup
import config
from dataset.dataset import *
from algorithms.analysisframework import *
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from algorithms.distance import *
from features.deep.test_net import *


#config_file = "/media/blcv/drive_2TB/CODE/FV-Benchmark/Nets/configs_net/casia_ver_conv52_felix.json"
#config      = parse_deep_config(config_file)
#data        = DataSet("deep", config )
#labels      = data.labels
#features    = data.loadFeatures().astype(np.float32)
#print "Read Featues"

##compute distances
#analysis = AnalysisFramework(config)
#supervised_learning = analysis.loadSupervisedAlgorithm('svm')

#load pairs
data = np.load('verification_felix.npy')
print "Read Pairs"
all_pairs = len(data)
step = 0.01
name_scores = "_data.npy"

number_of_parts = 100

for i in range(number_of_parts):
    iter_num = i / float(number_of_parts)
    print "Iter: ", i, int(iter_num * all_pairs), int((iter_num+ step)*all_pairs), all_pairs
    data_i    = data[int(iter_num * all_pairs): int((iter_num+ step)*all_pairs)]
    path    = str(i) + name_scores
    np.save(path, data_i)
    