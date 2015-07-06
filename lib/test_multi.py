import os, sys
import cv2
import config
from dataset.dataset import *
from algorithms.analysisframework import *
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from algorithms.distance import *
from features.deep.test_net import *
import time


'''Test BLUFR dataset with given configuration file. Firstly, all Algorithm need to be learned'''
# config_file = "/media/blcv/44488cdd-c584-4aab-9706-6929f09b9871/CODE/FV-Benchmark/Nets/configs_net/casia_ver_conv52.json"



def get_data():
  config_file = "/media/blcv/44488cdd-c584-4aab-9706-6929f09b9871/CODE/FV-Benchmark/Nets/configs_net/deepid2_fs.json"
  config = parse_deep_config(config_file)
  data = DataSet("deep", config )
  
  
  #Read LFW data
  features_lfw, label_ver_blufr = data.loadBLUFR()
  data_ver = load_cPickle(label_ver_blufr[0])
  #print "data readed"
  return data, data_ver, features_lfw

  #labels_val_ver = data.loadDataVerification('train')
  #labels_val_ver = data.loadDataClassification('val')
  #features = data.loadFeatures().astype(np.float32)
  
  #return data, labels_val_ver, features

def calc_acc_multi(data_ver, features):
  analysis = AnalysisFramework(config)
  time1 = time.time()
  acc, roc = analysis.computeStatsMulti(features, data_ver)
  time2 = time.time()
  print '%s function took %0.3f ms' % ("Multi", (time2-time1)*1000.0)
  print "ACC", acc
  

def calc_acc(data_ver, features):
  analysis = AnalysisFramework(config)
  time1 = time.time()
  acc, roc = analysis.computeStats(features, data_ver)
  time2 = time.time()
  print '%s function took %0.3f ms' % ("Single", (time2-time1)*1000.0)
  print "ACC", acc
  
  
  
if __name__ == '__main__':  
  data, label, features = get_data()
  calc_acc_multi(label, features)





