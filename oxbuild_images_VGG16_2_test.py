import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import json
import os
import progressbar 
import numpy as np

import Config_oxbuild_images_2 as config
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
from CropPreprocessor import CropPreprocessor
from RGBMeanSubtractionPreprocessor import RGBMeanSubtractionPreprocessor
from TrainingMonitor import TrainingMonitor
from HDF5DatasetGenerator import HDF5DatasetGenerator
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImageAspectPreprocessor import ImageAspectPreprocessor
from ImageDatasetLoader import ImageDatasetLoader
from FullyConnectedHead import FullyConnectedHead
from rank5_accuracy import rank5_accuracy


batch_size=4
class_num=config.NUM_CLASSES
MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "VGG16_oxbuild_images.model"])
TRAINED_HEAD_MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "VGG16_oxbuild_images_trained_head.model"])

means = json.loads(open(config.DATASET_RGBMEAN).read())

sp = ImagePreprocessor(224, 224)
mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()

print("[INFO] loading model...")
model = load_model(MODEL_PATH)
trainedHeadModel=load_model(TRAINED_HEAD_MODEL_PATH)
# print(model.summary())

print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size, preprocessors=[mp, sp, itap], classes=class_num)
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // batch_size, max_queue_size=batch_size * 2)
predictions1 = trainedHeadModel.predict_generator(testGen.generator(), steps=testGen.numImages // batch_size, max_queue_size=batch_size * 2)

(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-1: {:.2f}%".format(rank1 * 100))
(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-5: {:.2f}%".format(rank5 * 100))
testGen.close()

cp = CropPreprocessor(224, 224)
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size, preprocessors=[mp], classes=class_num)
predictions = []
predictions1 = []

widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // batch_size, widgets=widgets).start()

for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([itap.preprocess(c) for c in crops], dtype="float32")
        
        pred = model.predict(crops)
        pred1 = trainedHeadModel.predict(crops)
        predictions.append(pred.mean(axis=0))
        predictions1.append(pred1.mean(axis=0))
   
    pbar.update(i)
    
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-1: {:.2f}%".format(rank1 * 100))
(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-5: {:.2f}%".format(rank5 * 100))
testGen.close()