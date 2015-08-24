# Make sure that caffe is on the python path:
caffe_root = '/home/blcv/CODE/Caffe_NVIDIA/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

#Convert mean file produced by Caffe to numpy array
#python bin_to_npy.py dim_image_mean path_to_mean_caffe name_output


a=caffe.io.caffe_pb2.BlobProto()
file=open(sys.argv[2],'rb')
data = file.read()
a.ParseFromString(data)
means=a.data
means=np.asarray(means)
print means.shape
s = int(sys.argv[1])
means=means.reshape(3,s,s)
np.save(sys.argv[3],means)