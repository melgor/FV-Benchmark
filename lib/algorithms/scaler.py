#FV-Benchmark
import numpy as np
from copy import deepcopy

class Scaler(object):
  '''Scaler of data to desire range. Remember scale value for each feature, when traning to apply when testing'''
  def __init__(self, range_value):
    self.range_value = range_value
   
  '''Apply scaling to each feture independly'''
  def fit(self, data):
    self.list_max_min = list()
    self.num_features = data.shape[1]
    #find min and max value for every feature
    for idx in range(self.num_features):
      feat = data[:,idx]
      min_value = np.amin(feat)
      max_value = np.amax(feat)
      if max_value == min_value:
        self.list_max_min.append((0,1,1))
      else:
        self.list_max_min.append(( min_value, max_value, max_value - min_value))
      
  '''With found scale parameter, transform data to desire range'''    
  def transform(self, data):
    data_trans = deepcopy(data)
    for idx in range(self.num_features):
      feat = data[:,idx]
      data_trans[:, idx] = (feat -  self.list_max_min[idx][0])/ (self.list_max_min[idx][2])
      
    return data_trans
  
  
    
       
       
     
     
