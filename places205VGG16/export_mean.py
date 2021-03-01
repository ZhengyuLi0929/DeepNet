import sys
sys.path.append('/Users/apple/caffe/python')
import caffe
import numpy as np

#MEAN_PROTO_PATH = 'places205CNN_mean.binaryproto'
MEAN_NPY_PATH = 'places205VGG_mean.npy'

#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open(MEAN_PROTO_PATH, 'rb' ).read()
#blob.ParseFromString(data)
#
#array = np.array(caffe.io.blobproto_to_array(blob))
#mean_npy = array[0]
mean_npy = np.ones([3,256, 256], dtype=np.float)
mean_npy[0,:,:] = 105.487823486
mean_npy[1,:,:] = 113.741088867
mean_npy[2,:,:] = 116.060394287
np.save(MEAN_NPY_PATH ,mean_npy)
