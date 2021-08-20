import Config_oxbuild_images_1 as config
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

imagesPath = "datasets/oxbuild_images/images_train"
trainPaths=[]
trainLabels=[]

labels = os.listdir(imagesPath)
for label in labels:
    for image in paths.list_images(os.path.join(imagesPath,label,"good")):
        trainPaths.append(image)
        trainLabels.append(label)
    for image in paths.list_images(os.path.join(imagesPath,label,"ok")):
        trainPaths.append(image)
        trainLabels.append(label)

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=0.2, stratify=trainLabels, random_state=10)
(trainPaths, testPaths, trainLabels, testLabels) = split

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