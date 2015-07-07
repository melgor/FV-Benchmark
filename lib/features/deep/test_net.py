#FV-Benchmark
import numpy as np
from dataset.dataset import *
from extractor import *
from sklearn.metrics import accuracy_score

'''
Test net (Identification) which is set in config with given DataSet. 
WARNING: The Net have to be learned on same datset (Same number of labels)
'''
def testNet(config):
  config       = parse_deep_config(config)
  dataset      = DataSet("deep",config)
  extractor    = Extractor(config, config["caffe_proto_path_deploy"])
  path_data     = dataset.loadPathRawData()
  predictions = []
  for input_file in path_data:
    data = np.load(input_file)
    pred = extractor.predict_data(data)
    predictions.append(pred)
  
  
  predictions = np.vstack(predictions)
  y_pred_label = np.argmax(predictions, axis=1)

  #Get accuracy of training and validation set independly
  label_train = dataset.loadDataClassification("train")
  label_val   = dataset.loadDataClassification("val")

  #train
  predictions_train = y_pred_label[label_train]
  label_train       = dataset.labels[label_train]

  #val
  predictions_val = y_pred_label[label_val]
  label_val       = dataset.labels[label_val]

  print dataset.labels, y_pred_label
  print "Accuracy Train: ", accuracy_score(label_train, predictions_train)
  print "Accuracy Val: ", accuracy_score(label_val, predictions_val)
  
def testNetFromFile(path_to_file, config):

  config       = parse_deep_config(config)
  extractor    = Extractor(config, config["caffe_proto_path_deploy"])
  files = []
  labels = []
  #read file
  with open(path_to_file, 'r') as f:
    for path in f:
      line = path.strip().split(' ')
      files.append(line[0])
      labels.append(int(line[1]))
  
  in_one_file = 128 
  current_num = 0
  predictions = []
  for i in range(0, len(files), in_one_file):
    if (i + in_one_file) < len(files):
      curr_list = files[i:i + in_one_file] 
    else:
      curr_list = files[i:] 
    pred = extractor.predict_multi(curr_list)
    pred = np.argmax(pred, axis=1)
    predictions.append(pred)
    current_num += in_one_file
    print "Predicted: ", current_num, " from ", len(files)
  
  y_pred_label = np.hstack(predictions)
  # y_pred_label = np.argmax(predictions, axis=1)
  labels       = np.asarray(labels)
  # print labels, y_pred_label
  print "Accuracy: ", accuracy_score(labels, y_pred_label)
  
  
      