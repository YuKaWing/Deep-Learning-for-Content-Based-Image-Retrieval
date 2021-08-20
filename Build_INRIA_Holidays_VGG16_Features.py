from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from HDF5FeatureWriter import HDF5FeatureWriter
from imutils import paths
import numpy as np
import progressbar
import os
import cv2
import h5py
import Config_INRIA_Holidays as config
from RGBMeanSubtractionPreprocessor import RGBMeanSubtractionPreprocessor
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
import json

imagePaths = list(paths.list_images(config.IMAGES_PATH))
outputPath = os.path.sep.join([config.HDF5_PATH, "features_VGG16.hdf5"])
bs = 20
modelPath = os.path.sep.join([config.OUTPUT_PATH, "VGG16_INRIA_Holidays.hdf5"])
means = json.loads(open(config.DATASET_RGBMEAN).read())

featureset = HDF5FeatureWriter((len(imagePaths), 512), outputPath, bufSize = bs)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
widgets=widgets).start()

model = load_model(modelPath)
x = model.layers[-2].output
model = Model(inputs = model.input, outputs = x)
print(model.summary())

ip = ImagePreprocessor(224, 224)
mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()


for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        image = cv2.imread(imagePath)
        image = image.astype('Float64')
        image = ip.preprocess(image)
        image = mp.preprocess(image)
        image = itap.preprocess(image)
        image = np.expand_dims(image, axis=0)
        batchImages.append(image)
    
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 512))
    featureset.add(features, batchPaths)
    pbar.update(i)

featureset.close()
pbar.finish()