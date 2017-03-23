import cv2
from matplotlib import pyplot as plt
import os                 
from random import shuffle
from tqdm import tqdm
import sqlite3
import tflearn
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph()

#fetch frames and labels from sqlite file
conn = sqlite3.connect('/Users/mikko/Documents/TireFiles/basler_1487949884000.avi.db')
c = conn.cursor()
c.execute('SELECT frameno, labelsjson FROM labels')
data = c.fetchall()

#set parameters
traindir = '/Users/mikko/Documents/TireFiles/TrainImgs'
testdir = '/Users/mikko/Documents/TireFiles/TestImgs'
imgsize = 32
learnrate = 1e-3

modelname = 'tirelearning-{}-{}.model'.format(learnrate, '6convlearn4')

#form lists from collected data
frames = [str(row[0]) for row in data]
labels = [str(row[1])[:-2][2:] for row in data]

#dictionary to classify frames
labelsdict = {
    'hole': [],
    'l5': []
}

#iterate through frames and add classify them inside labelsdict depending on the label
for i in range(len(labels)):
    #remove poor data
    if labels[i][-4:] != "poor":
        if labels[i] == "qahole":
            labelsdict['hole'].append(frames[i])
        elif labels[i] == "l5":
            labelsdict['l5'].append(frames[i])

#returns one-hot array for each frame depending on the label
def label_img(img):
    #return one-hot array depending on label
    if img in labelsdict['l5']:
        #[l5, not hole, not normal]
        return [1,0,0]
    
    elif img in labelsdict['hole']:
        #[not l5, hole, not normal]
        return [0,1,0]

    else:
        #[not l5, not hole, normal]
        return [0,0,1]

def create_train_data():
    training_data = []
    
    #list with data from traindir
    trainlist = os.listdir(traindir)
    
    #remove apple's default DS_Store from list
    del trainlist[0]

    #iterate through images
    for img in trainlist:
        #remove '.png' and leading zeros and create one-hot array
        imgindex = img[:-4].lstrip("0")
        label = label_img(imgindex)

        #create path to image
        path = os.path.join(traindir,img)

        #read image into grayscale and resize
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (imgsize,imgsize))

        #add data into training_data
        training_data.append([np.array(img),np.array(label)])
        
    #shuffle the data
    shuffle(training_data)

    #save and return
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    
    #list with data from traindir
    testlist = os.listdir(testdir)
    
    #remove apple's default DS_Store from list
    del testlist[0]

    #iterate through images
    for img in testlist:
        #remove '.png' and leading zeros
        imgindex = img[:-4].lstrip("0")

        #create path to image
        path = os.path.join(testdir,img)

        #read image into grayscale and resize
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (imgsize,imgsize))

        #add data into training_data
        testing_data.append([np.array(img),np.array(label)])
        
    #shuffle the data
    shuffle(testing_data)

    #save and return
    np.save('test_data.npy', testing_data)
    return testing_data

traindata = create_train_data()

convnet = input_data(shape=[None, imgsize, imgsize, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learnrate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(modelname)):
    model.load(modelname)
    print('model loaded!')

train = traindata[:-30]
test = traindata[-30:]

X = np.array([i[0] for i in train]).reshape(-1,imgsize,imgsize,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,imgsize,imgsize,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=modelname)
























