import Config_INRIA_Holidays as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ImagePreprocessor import ImagePreprocessor
from HDF5DatasetWriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
import random

tempPaths = list(paths.list_images(config.IMAGES_PATH))
trainPaths=[]
testPaths = []
trainLabels = []
testLabels = []

random.shuffle(tempPaths)

for tempPath in tempPaths:
    label = tempPath.split("\\")[-1][0:4]
    temp = tempPath.split("\\")[-1][4:]
    
    if temp=="00.jpg":
        testPaths.append(tempPath)
        testLabels.append(label)
    else:
        trainPaths.append(tempPath)
        trainLabels.append(label)

le = LabelEncoder()
temp1train = le.fit_transform(trainLabels)
temp1test = le.fit_transform(testLabels)

le.fit(trainLabels)
temp2train = le.transform(trainLabels)
temp2test = le.transform(testLabels)


random.shuffle(trainLabels)
le.fit(trainLabels)
temp3train = le.transform(trainLabels)
temp3test = le.transform(testLabels)

print(temp1test)
print(temp2test)
print(temp3test)