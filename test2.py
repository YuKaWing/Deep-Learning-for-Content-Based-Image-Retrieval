from imutils import paths
from shutil import copy2
import Config_oxbuild_images_2 as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ImagePreprocessor import ImagePreprocessor
from HDF5DatasetWriter import HDF5DatasetWriter
import numpy as np
import progressbar
import json
import cv2
import os
import random

txtsPath="datasets/oxbuild_images/gt_files_170407"

queries=[]
for txt in os.listdir(txtsPath):
    if txt.split('_1_')[-1] == "query.txt" or txt.split('_2_')[-1] == "query.txt":
        queries.append(txt.split("query.txt")[0])

queryImageIDs=[]

for query in queries:
    file = open(os.path.join(txtsPath, query+"query.txt"), "r")
    queryImageIDs.append((file.readline().split(' ')[0]).split('oxc1_')[-1])
    file.close()

imagesPath = "datasets/oxbuild_images/images_train"
trainPaths=[]
trainLabels=[]
testPaths=[]
testLabels=[]

labels = os.listdir(imagesPath)
for label in labels:
    for image in paths.list_images(os.path.join(imagesPath,label,"good")):
        if image.split(os.path.sep)[-1].split('.jpg')[0] in queryImageIDs:
            testPaths.append(image)
            testLabels.append(label)
        else:
            trainPaths.append(image)
            trainLabels.append(label)
    for image in paths.list_images(os.path.join(imagesPath,label,"ok")):
        if image.split(os.path.sep)[-1].split('.jpg')[0] in queryImageIDs:
            testPaths.append(image)
            testLabels.append(label)
        else:
            trainPaths.append(image)
            trainLabels.append(label)

train_set=list(zip(trainPaths,trainLabels))
test_set=list(zip(testPaths,testLabels))

random.shuffle(train_set)
random.shuffle(test_set)

trainPaths = [i for i, j in train_set]
trainLabels = [j for i, j in train_set]
testPaths = [i for i, j in test_set]
testLabels = [j for i, j in test_set]

for testPath in testPaths:
    copy2(testPath,"datasets\\oxbuild_images\\images_query")