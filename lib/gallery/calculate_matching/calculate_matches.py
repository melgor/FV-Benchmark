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

# Run Parallel calculating matching using bash command (better than using Python MutliCore)
# ls *_data.npy | parallel python calculate_matches.py 
print "read: ", sys.argv[1]
data_pairs = np.load(sys.argv[1])

config_file = "/media/blcv/drive_2TB/CODE/FV-Benchmark/Nets/configs_net/casia_ver_conv52_correct_felix.json"
config      = parse_deep_config(config_file)
data        = DataSet("deep", config )
labels      = data.labels
features    = data.loadFeatures().astype(np.float32)
print "Read Featues"

#compute distances
analysis = AnalysisFramework(config)
supervised_learning = analysis.loadSupervisedAlgorithm('svm')

label_ver = (data_pairs, list())

X =  supervised_learning.scale.transform(features)
match, mis = computeDistanceMatrix(X, label_ver, supervised_learning.distance)
#match, mis = threadComputeMatrix(X, label_ver, supervised_learning.distance)
data_match = supervised_learning.clf.decision_function( np.asarray(match) )
path = "scores_" + sys.argv[1]
np.save(path, data_match)