import Config_oxbuild_images_2 as config
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

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]

# datasets = [
#     ("train", trainPaths, trainLabels, config.TRAIN_HDF5)
#     ]

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