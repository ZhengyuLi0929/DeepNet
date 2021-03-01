import sys
sys.path.append('/Users/apple/caffe/python')

import caffe
import json
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
'''
Bedroom = '../Scene Images/Good categories/Bedroom/'
Dining = '../Scene Images/Good categories/Dining/'
Forest = '../Scene Images/Good categories/Forest/'
Hospital = '../Scene Images/Good categories/Hospital/'
Mountain = '../Scene Images/Good categories/Mountain/'
Office = '../Scene Images/Good categories/Office/'
'''

Beach_good = '../Scene Images/fall 20/Scenes/good_beaches/'
Forest_good = '../Scene Images/fall 20/Scenes/good_forests/'
Highway_good = '../Scene Images/fall 20/Scenes/good_highways/'
Mountain_good = '../Scene Images/fall 20/Scenes/good_mountains/'
Office_good = '../Scene Images/fall 20/Scenes/good_offices/'
Beach_medium = '../Scene Images/fall 20/Scenes/medium_beaches/'
Forest_medium = '../Scene Images/fall 20/Scenes/medium_forests/'
Highway_medium = '../Scene Images/fall 20/Scenes/medium_highways/'
Mountain_medium = '../Scene Images/fall 20/Scenes/medium_mountains/'
Office_medium = '../Scene Images/fall 20/Scenes/medium_offices/'
Beach_bad = '../Scene Images/fall 20/Scenes/bad_beaches/'
Forest_bad = '../Scene Images/fall 20/Scenes/bad_forests/'
Highway_bad = '../Scene Images/fall 20/Scenes/bad_highways/'
Mountain_bad = '../Scene Images/fall 20/Scenes/bad_mountains/'
Office_bad = '../Scene Images/fall 20/Scenes/bad_offices/'

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



#directories = [Bedroom, Dining, Forest, Hospital, Mountain, Office]
directories = [Beach_good, Forest_good, Highway_good, Mountain_good, Office_good,
               Beach_medium, Forest_medium, Highway_medium, Mountain_medium, Office_medium,
               Beach_bad, Forest_bad, Highway_bad, Mountain_bad, Office_bad]
matrix = []
json_good = {}
json_medium = {}
json_bad = {}
ndir = 0
for directory in directories:
    conf_arr = np.zeros(5)
    count = 0
    for filename in os.listdir(directory):
        if ndir < 5:
            json_good[filename] = []
        elif ndir < 10:
            json_medium[filename] = []
        else:
            json_bad[filename] = []
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            count += 1
            imgpath = os.path.join(directory, filename)
            input_image = caffe.io.load_image(imgpath)
            net.blobs['data'].data[...] = transformer.preprocess('data',input_image)
            out = net.forward()
            
            prediction = net.blobs['prob'].data[0].flatten()
            '''
            conf_arr[0] += prediction[24]
            conf_arr[1] += prediction[64]
            conf_arr[2] += max(np.array([prediction[16], prediction[78], prediction[79], prediction[149]]))
            conf_arr[3] += max(np.array([prediction[94], prediction[95]]))
            conf_arr[4] += max(np.array([prediction[121], prediction[122]]))
            conf_arr[5] += max(np.array([prediction[93], prediction[129], prediction[130]]))
            '''
            conf_arr[0] += prediction[47]
            conf_arr[1] += max(np.array([prediction[16], prediction[78], prediction[79], prediction[149]]))
            conf_arr[2] += prediction[92]
            conf_arr[3] += max(np.array([prediction[121], prediction[122]]))
            conf_arr[4] += max(np.array([prediction[93], prediction[129], prediction[130]]))
            if ndir < 5:
                json_good[filename].append({'beach': conf_arr[0], 'forest': conf_arr[1], 'highway': conf_arr[2],
                    'mountain': conf_arr[3], 'office': conf_arr[4]})
            elif ndir < 10:
                json_medium[filename].append({'beach': conf_arr[0], 'forest': conf_arr[1], 'highway': conf_arr[2],
                    'mountain': conf_arr[3], 'office': conf_arr[4]})
            else:
                json_bad[filename].append({'beach': conf_arr[0], 'forest': conf_arr[1], 'highway': conf_arr[2],
                    'mountain': conf_arr[3], 'office': conf_arr[4]})
    conf_arr = conf_arr / count
    conf_arr = np.around(conf_arr, decimals = 12)
    conf_arr = conf_arr.tolist()
    matrix.append(conf_arr)
    ndir += 1

'''
print("\t \t Bedroom \t Dining \t Forest \t Hospital \t Mountain \t Office")
print("Bedroom",list(np.around(matrix[0], decimals = 10 )))
print("Dining", list(np.around(matrix[1], decimals = 10 )))
print("Forest", list(np.around(matrix[2], decimals = 10 )))
print("Hospital", list(np.around(matrix[3], decimals = 10 )))
print("Mountain", list(np.around(matrix[4], decimals = 10 )))
print("Office", list(np.around(matrix[5], decimals = 10 )))
'''

print("mt1 = np.array("+str(list(np.around(matrix[0], decimals = 10 )))+")")
print("mt2 = np.array("+str(list(np.around(matrix[1], decimals = 10 )))+")")
print("mt3 = np.array("+str(list(np.around(matrix[2], decimals = 10 )))+")")
print("mt4 = np.array("+str(list(np.around(matrix[3], decimals = 10 )))+")")
print("mt5 = np.array("+str(list(np.around(matrix[4], decimals = 10 )))+")")

print("mt6 = np.array("+str(list(np.around(matrix[5], decimals = 10 )))+")")
print("mt7 = np.array("+str(list(np.around(matrix[6], decimals = 10 )))+")")
print("mt8 = np.array("+str(list(np.around(matrix[7], decimals = 10 )))+")")
print("mt9 = np.array("+str(list(np.around(matrix[8], decimals = 10 )))+")")
print("mt10 = np.array("+str(list(np.around(matrix[9], decimals = 10 )))+")")

print("mt11 = np.array("+str(list(np.around(matrix[10], decimals = 10 )))+")")
print("mt12 = np.array("+str(list(np.around(matrix[11], decimals = 10 )))+")")
print("mt13 = np.array("+str(list(np.around(matrix[12], decimals = 10 )))+")")
print("mt14 = np.array("+str(list(np.around(matrix[13], decimals = 10 )))+")")
print("mt15 = np.array("+str(list(np.around(matrix[14], decimals = 10 )))+")")

with open('normal_good.json', 'w') as outfile1:
    json.dump(json_good, outfile1)
with open('normal_medium.json', 'w') as outfile2:
    json.dump(json_medium, outfile2)
with open('normal_bad.json', 'w') as outfile3:
    json.dump(json_bad, outfile3)
    
