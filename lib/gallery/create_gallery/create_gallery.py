import os
import cv2
import startup
import config
from dataset.dataset import DataSet
from algorithms.analysisframework       import AnalysisFramework
from algorithms.distance import *
from utils import *

  
#Read data associated with Net
label_file = "/media/blcv/drive_2TB/CODE/FV-Benchmark/DataSets/Felix/labels.txt"
main_folder = "/media/blcv/drive_2TB/CODE/FV-Benchmark/"
config_file = main_folder + "Nets/configs_net/casia_ver_conv52_correct_felix.json"
config_file = parse_deep_config(config_file)
data = DataSet("deep",config_file)
labels  = data.labels
labels_set = set(labels)

#After calulcating matches outside iPython (using parallel etc.), load score and extract best images
score_file = "/media/blcv/drive_2TB/CODE/FV-Benchmark/FELIX_data/calculate_matching/divide_pairs/scores_all.txt"
with open(score_file, 'r') as f:
    lines_score = [line.strip() for line in f]
lines_score = sorted(lines_score,key=lambda x: int(x.split('/')[-1].split('.')[0].split("_")[1]),  reverse=False)

verification_task = np.load("/media/blcv/drive_2TB/CODE/FV-Benchmark/notebooks/verification_felix.npy")

scores_matrix = np.zeros((verification_task.shape[0]))
start_elem = 0
for idx, elem in enumerate(lines_score):
    data = np.load(elem)
    scores_matrix[start_elem: start_elem + data.shape[0]] = data
    start_elem += data.shape[0]
    
with open(label_file, 'r') as f:
    label_file_raw = [line.strip() for line in f]
    
#acculumate scores per class per instance
accumulate_score = dict()
for elem in labels_set:
    accumulate_score[str(elem)] = dict()

num = 0    
for elem, score in zip(verification_task, scores_matrix):
    print num
    num += 1
    idx = elem[0]
    label_idx = labels[idx]
    if str(idx) not in  accumulate_score[str(label_idx)].keys():
      accumulate_score[str(label_idx)][str(idx)] = 0
    accumulate_score[str(label_idx)][str(idx)] += score
    
  
        
save_cPickle("acc_score.dict", accumulate_score)