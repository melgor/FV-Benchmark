import numpy as np
import os

# Make sure that caffe is on the python patht this file is expected to be in {caffe_root}/examples
caffe_root = '/home/blcv/LIB/caffe_master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))


class Extractor(object):
  """docstring for Prediction_Normalme"""
  def __init__(self, config_net, proto_path):
    # Set the right path to your model definition file, pretrained model weights.
    self.net = caffe.Classifier (
              proto_path,
              config_net["caffe_net_path"],
              mean         = np.load(config_net["caffe_mean_image"]),
              raw_scale    = 255,
              image_dims   = config_net["caffe_image_size"],
              channel_swap = (2,1,0))        
    caffe.set_mode_gpu()
    caffe.set_device(config_net["caffe_gpu_id"])  
    
  def predict(self, image):
    """Predict using Caffe normal model"""
    input_image = caffe.io.load_image(image,color=False)
    prediction = self.net.predict([input_image], oversample=False)
    return prediction.astype(np.float16)
      
  def predict_multi(self, images):
    """Predict using Caffe normal model"""
    list_input = list()
    for image in images: 
      list_input.append(caffe.io.load_image(image,color=True))
    prediction = self.net.predict(list_input, oversample=False)
    return prediction.astype(np.float16)
  
  def predict_data(self,data):
    #d2 = np.swapaxes(data,3,1)
    #d2 = np.swapaxes(d2,3,2)
    list_images = [img for img in data]
    prediction = self.net.predict(list_images, oversample=False)
    return prediction.astype(np.float16)
  

#class Extractor(caffe.Net):

    #"""docstring for Prediction_Normalme"""

    #def __init__(self, proto_path, bin_path):
        ## Set the right path to your model definition file, pretrained model weights,
        ## and the image you would like to classify.
        #MODEL_FILE = proto_path
        #PRETRAINED = bin_path

        ##caffe.set_phase_test()
        #caffe.set_device(0)
        #caffe.set_mode_gpu()
        ##caffe.set_device(1)
        #caffe.Net.__init__(self, MODEL_FILE, PRETRAINED, caffe.TEST)

        ##self.set_raw_scale(self.inputs[0], 255.0)
        ##self.set_mean(self.inputs[0], np.load('mean_file.npy'))
        ##self.image_dims  = (45, 45)
        ##self.crop_dims = np.array(self.blobs[self.inputs[0]].data.shape[2:])
        
         ## configure pre-processing
        #in_ = self.inputs[0]
        #self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        #self.transformer = caffe.io.Transformer(
            #{in_: self.blobs[in_].data.shape for in_ in self.inputs})
        #self.transformer.set_transpose(in_, (2,0,1))
        #self.transformer.set_mean(in_, np.load('mean_file_id.npy'))
        #self.transformer.set_raw_scale(in_, 255.0)
        #self.transformer.set_channel_swap(in_, (2,1,0))
        
        #self.image_dims = (45, 45)
    
    #def predict_multi(self, images):
        #"""
        #Predict classification probabilities of inputs.

        #Take
        #inputs: iterable of (H x W x K) input ndarrays.
        #oversample: average predictions across center, corners, and mirrors
                    #when True (default). Center-only prediction when False.

        #Give
        #predictions: (N x C) ndarray of class probabilities
                     #for N images and C classes.
        #"""
        ## Scale to standardize input dimensions.
        #list_input = list()
        #for image in images:
            #list_input.append(caffe.io.load_image(image, color=True))
            
        #input_ = np.zeros((len(list_input),
                           #self.image_dims[0], self.image_dims[1], list_input[0].shape[2]),
                          #dtype=np.float32)    
  
        #for ix, in_ in enumerate(list_input):
            #input_[ix] = caffe.io.resize_image(in_, self.image_dims)

       
        ## Take center crop.
        #center = np.array(self.image_dims) / 2.0
        #crop = np.tile(center, (1, 2))[0] + np.concatenate([
            #-self.crop_dims / 2.0,
            #self.crop_dims / 2.0
        #])
        #input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        ## Classify
        #caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            #dtype=np.float32)
        #for ix, in_ in enumerate(input_):
            #caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        #out = self.forward_all(**{self.inputs[0]: caffe_in})
        #predictions = out[self.outputs[0]].squeeze(axis=(2,3))

       
        #return predictions.astype(np.float16)
      
    ##def predict_multi(self, images):
        ##"""Predict using Caffe normal model"""
        

        ##input_ = np.zeros((len(list_input),
                           ##self.image_dims[0], self.image_dims[1], list_input[0].shape[2]),
                          ##dtype=np.float32)
        ##for ix, in_ in enumerate(list_input):
            ##input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        ### Take center crop.
        ##center = np.array(self.image_dims) / 2.0
        ##crop = np.tile(center, (1, 2))[0] + np.concatenate([
            ##-self.crop_dims / 2.0,
            ##self.crop_dims / 2.0
        ##])
        ##input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        ### run network
        ##caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            ##dtype=np.float32)
        ##for ix, in_ in enumerate(input_):
            ##caffe_in[ix] = self.preprocess(self.inputs[0], in_)
        ##out = self.forward_all(**{self.inputs[0]: caffe_in})
        ##predictions = out[self.outputs[0]].squeeze(axis=(2, 3))

        ##return predictions.astype(np.float16)
