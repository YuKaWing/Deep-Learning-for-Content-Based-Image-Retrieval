import Config_256_ObjectCategoriess as config
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
for tempPath in tempPaths:
    temp = tempPath.split("_")[-1]
    if tempPath.split(os.path.sep)[-2] != "257.clutter":
        if temp=="0001.jpg" or temp=="0002.jpg" or temp=="0003.jpg" or temp=="0004.jpg" or temp=="0005.jpg":
            testPaths.append(tempPath)
        else:
            trainPaths.append(tempPath)

random.shuffle(trainPaths)
random.shuffle(testPaths)
    
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
testLabels = [p.split(os.path.sep)[-2] for p in testPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]

ip = ImagePreprocessor(224, 224)
(R, G, B) = ([], [], [])
meanImage=np.zeros((224, 224, 3));

for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 224, 224, 3), outputPath)
    writer.storeClassLabels(le.classes_)
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()   
   
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = ip.preprocess(image)
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
            
            meanImage+=image;
       
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()
    
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_RGBMEAN, "w")
f.write(json.dumps(D))
f.close()

meanImage = meanImage/len(trainPaths)
np.save(config.DATASET_IMAGEMEAN, meanImage)