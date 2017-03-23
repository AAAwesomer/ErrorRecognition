import cv2
from matplotlib import pyplot as plt
import os                 
from random import shuffle
from tqdm import tqdm
import sqlite3
import tflearn
import numpy as np

#fetch frames and labels from sqlite file
conn = sqlite3.connect('/Users/mikko/Documents/AffectoFiles/basler_1487949884000.avi.db')
c = conn.cursor()
c.execute('SELECT frameno, labelsjson FROM labels')
data = c.fetchall()

#set parameters
traindir = '/Users/mikko/Documents/AffectoFiles/TrainImgs'
testdir = '/Users/mikko/Documents/AffectoFiles/TestImgs'
imgsize = 80
LR = 1e-3

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

##        ##test:
##        plt.imshow(img, cmap='gray', interpolation='bicubic')
##        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
##        plt.show()
##        break

        #add data into training_data
        training_data.append([np.array(img),np.array(label)])
        
    #shuffle the data
    shuffle(training_data)

    #save and return
    np.save('train_data.npy', training_data)
    return training_data

##traindata = create_train_data()





















