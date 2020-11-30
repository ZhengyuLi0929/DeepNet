import sys
sys.path.append('/Users/apple/caffe/python')

import caffe

import numpy as np
import scipy


# caffe model
caffe.set_mode_cpu()

# protxt
model_def = './deploy_places205.protxt'

# load model
model_weights = './googlelet_places205_train_iter_2400000.caffemodel'

# mean
mean_file = './googlenet_mean.npy'

# read caffenet
#net = caffe.Net(model_def,      # defines the structure of the model
#                model_weights,  # contains the trained weights
#                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#net = caffe.Classifier(model_def, model_weights)

net = caffe.Net(model_def,model_weights,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

Bedroom = '../Scene Images/Good categories/Bedroom/'
Dining = '../Scene Images/Good categories/Dining/'
Forest = '../Scene Images/Good categories/Forest/'
Hospital = '../Scene Images/Good categories/Hospital/'
Mountain = '../Scene Images/Good categories/Mountain/'
Office = '../Scene Images/Good categories/Office/'

import os
#img1 = os.path.join(Bedroom, 'bedroom2.jpeg')
#img2 = os.path.join(Bedroom, 'bedroom3.jpg')
#ipt1 = caffe.io.load_image(img1)
#ipt2 = caffe.io.load_image(img2)
#
#
##net.blobs['data'].data[...] = transformer.preprocess('data',ipt1)
##out = net.forward()
##
##print(net.blobs['prob'].data[0].flatten())
#
#net.blobs['data'].data[...] = transformer.preprocess('data',ipt2)
#out = net.forward()
#
#res =net.blobs['prob'].data[0].flatten()
#print(res)
#print(len(res))
#maxid = max(res)
#print(maxid)
#print(res[24])


#pre1 = net.predict([ipt1], oversample = False)
#pre2 = net.predict([ipt2], oversample = False)
#print (pre1, pre2)



directories = [Bedroom, Dining, Forest, Hospital, Mountain, Office]
matrix = []
for directory in directories:
    conf_arr = np.zeros(6)
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            count += 1
            imgpath = os.path.join(directory, filename)
            input_image = caffe.io.load_image(imgpath)
            net.blobs['data'].data[...] = transformer.preprocess('data',input_image)
            out = net.forward()
            
            prediction = net.blobs['prob'].data[0].flatten()
        
            conf_arr[0] += prediction[24]
            conf_arr[1] += prediction[64]
            conf_arr[2] += max(np.array([prediction[16], prediction[78], prediction[79], prediction[149]]))
            conf_arr[3] += max(np.array([prediction[94], prediction[95]]))
            conf_arr[4] += max(np.array([prediction[121], prediction[122]]))
            conf_arr[5] += max(np.array([prediction[93], prediction[129], prediction[130]]))

    conf_arr = conf_arr / count
    conf_arr = np.around(conf_arr, decimals = 12)
    conf_arr = conf_arr.tolist()
    matrix.append(conf_arr)


print("\t \t Bedroom \t Dining \t Forest \t Hospital \t Mountain \t Office")
print("Bedroom",list(np.around(matrix[0], decimals = 10 )))
print("Dining", list(np.around(matrix[1], decimals = 10 )))
print("Forest", list(np.around(matrix[2], decimals = 10 )))
print("Hospital", list(np.around(matrix[3], decimals = 10 )))
print("Mountain", list(np.around(matrix[4], decimals = 10 )))
print("Office", list(np.around(matrix[5], decimals = 10 )))
