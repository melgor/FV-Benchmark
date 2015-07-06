import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.lda import LDA
from verification import *
from learning.pca import *
from learning.svm import *
from learning.joint_bayesian import *
from scaler import *
import os
execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config
from utils import *
from collections import namedtuple
Commpresion_obj = namedtuple('Commpresion_obj', 'algorithm whitening')

class AnalysisFramework:

    def __init__(self, config_net, descriptors = None):
      self.compressed_descriptors = None
      self.config_net             = config_net
      self.descriptors            = descriptors

    def computeStats(self, descriptors, ground_truth,   distance = cosineDistance, scaler = False, reset_scaler = False):
      if scaler:
        descriptors = self.scaleData(descriptors, reset_scaler)
      
      scores = computeDistanceMatrix(descriptors, ground_truth, distance)
      acc,roc    = computeAccuracyROC(scores)
      # roc    = computeROC(scores)
      return acc, roc
    
    def computeStatsMulti(self, descriptors, ground_truth,   distance = cosineDistance, scaler = False, reset_scaler = False):
      if scaler:
        descriptors = self.scaleData(descriptors, reset_scaler)
      
      scores  = threadComputeMatrix(descriptors, ground_truth, distance)
      acc,roc = computeAccuracyROC(scores)
      thres   = getBestThreshold(scores)
      print "Best Thres:" , thres
      return acc, roc

    def displayStats(self, labels, accs, rocs, name_set =  None):
      for label, acc in zip(labels, accs):
        print "%s: %0.4f"%(label, acc)

      plotROC(rocs, labels, "roc")
      if name_set != None:
        #save result for other manipulation
        results     = (labels, accs, rocs)
        path_result = os.path.join(config.models_path,  self.config_net['name'], name_set +'.npy')
        save_cPickle(results, path_result)

    #scale all features to be between 0-1        
    def scaleData(self, data, reset_scaler):
      if reset_scaler:
        self.scaler = Scaler((0,1))
        self.scaler.fit(data)
        self.saveSupervisedAlgorithm('scaler', self.scaler)
      else:
        self.scaler = self.loadSupervisedAlgorithm('scaler')
      return self.scaler.transform(data)

    def computeDescriptors(self, descriptor_func, data):
      if self.descriptors is not None:
        del self.descriptors
        self.descriptors = None
          
      self.descriptors = descriptor_func(data)
      return self.computeStatsMulti(self.descriptors)
  
    
    def compressDescriptors(self, method, training_data, ground_truth, dim, distance = cosineDistance):
      if self.compressed_descriptors is not None:
        del self.compressed_descriptors
        self.compressed_descriptors = None
      
      if method == "pca":
        self.compression = computeProbabilisticPCA(training_data, dim=dim)
        self.whitening_       = np.diag(np.power(self.compression.explained_variance_, -0.5))
      elif method == "rp":
        X_               = lil_matrix((len(self.descriptors), self.descriptors.shape[1]))
        self.compression = GaussianRandomProjection(n_components=dim)
        self.compression.fit(X_)
        self.whitening_       = np.eye(dim)
      elif method == "srp":
        X_               = lil_matrix((len(self.descriptors), self.descriptors.shape[1]))
        self.compression = SparseRandomProjection(n_components=dim, dense_output=True)
        self.compression.fit(X_)
        self.whitening_       = np.eye(dim)
      else:
        raise Exception("Compression method unknown")
      
      compressed_descriptors = self.compression.transform(self.descriptors)
      for i in xrange(len(compressed_descriptors)):
        compressed_descriptors[i] = np.dot(self.whitening_, compressed_descriptors[i])

      comp_obj = Commpresion_obj(self.compression, self.whitening_)
      self.saveSupervisedAlgorithm(method, comp_obj)
      return self.computeStatsMulti(compressed_descriptors, ground_truth, distance)
    
    def compressDescriptorsTransform(self, data, ground_truth, name, distance = cosineDistance):
      comp_obj= self.loadSupervisedAlgorithm(name)
      compressed_descriptors = comp_obj.algorithm.transform(data) 
      for i in xrange(len(compressed_descriptors)):
          compressed_descriptors[i] = np.dot(comp_obj.whitening, compressed_descriptors[i])
      return self.computeStatsMulti(compressed_descriptors, ground_truth, distance)
    
    def supervisedLearningLDA(self, training_data, ground_truth ,dim=50, distance = cosineDistance):
      self.supervised_learning = LDA(dim)        
      self.supervised_learning.fit(training_data[0], training_data[1])
      self.saveSupervisedAlgorithm('lda', self.supervised_learning)
      descriptors_ = self.supervised_learning.transform(self.descriptors)
      return self.computeStatsMulti(descriptors_,ground_truth, distance = distance)

    def supervisedLearningPredictLDA(self, data, ground_truth, distance = cosineDistance):
      self.supervised_learning = self.loadSupervisedAlgorithm('lda')
      descriptors_ = self.supervised_learning.transform(data)
      return self.computeStatsMulti(descriptors_,ground_truth)

    def supervisedLearningJB(self, training_data, ground_truth):
      self.supervised_learning = JointBayesian()
      self.supervised_learning.fit(training_data[0], training_data[1])
      self.saveSupervisedAlgorithm('jb', self.supervised_learning)
      return self.computeStatsMulti(self.descriptors, ground_truth, distance = self.supervised_learning.mesureDistance)

    def supervisedLearningPredictJB(self, data, ground_truth):
      self.supervised_learning = self.loadSupervisedAlgorithm('jb')
      return self.computeStatsMulti(data, ground_truth, distance = self.supervised_learning.mesureDistance)

    def supervisedLearningSVM(self, training_data, distance):
      self.supervised_learning = SVM(training_data[1])
      self.supervised_learning.fit(training_data[0], training_data[1], distance)  
      self.saveSupervisedAlgorithm('svm', self.supervised_learning)
      scores_ = self.supervised_learning.transform(self.descriptors, training_data[1] )
      acc,roc    = computeAccuracyROC(scores_)
      thres   = getBestThreshold(scores_)
      print "Best Thres:" , thres
      # roc    = computeROC(scores_)
      return  acc, roc

    def supervisedLearningPredictSVM(self, data, ground_truth):
      self.supervised_learning = self.loadSupervisedAlgorithm('svm')
      scores_ = self.supervised_learning.transform(data, ground_truth )
      acc,roc    = computeAccuracyROC(scores_)
      thres   = getBestThreshold(scores_)
      print "Best Thres:" , thres
      # roc    = computeROC(scores_)
      return  acc, roc

    def saveSupervisedAlgorithm(self, name, object_to_save):
      '''Save algorith to reuse it. Name based on algoritmh and configuration of data'''
      print "Save algorithm: ", name
      path_algorithm = os.path.join(config.models_path,  self.config_net['name'], name +'.npy')
      create_dir(os.path.dirname(path_algorithm))
      save_cPickle(object_to_save, path_algorithm)

    def loadSupervisedAlgorithm(self, name):
      path_algorithm = os.path.join(config.models_path,  self.config_net['name'], name +'.npy')
      return load_cPickle(path_algorithm)
    

    def showFeaturesStatistic(self, features):
      mean_value = np.mean(features.astype(np.float32), axis =0 )
      std        = np.std(features.astype(np.float32), axis =0 )
      ind = np.arange(features.shape[1])  # the x locations for the groups
      width = 3.0       # the width of the bars
      fig, ax = plt.subplots()
      rects1 = ax.bar(ind, mean_value, width, color='r')#, yerr=std)
      plt.show()



    # def computeStatsBLUFR(self, descriptors):
      