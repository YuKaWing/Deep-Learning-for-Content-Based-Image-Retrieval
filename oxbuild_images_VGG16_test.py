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

import Config_oxbuild_images as config
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
from PatchPreprocessor import PatchPreprocessor
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
TRAINED_HEAD_MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "VGG16_oxbuild_images_trained_head"])

means = json.loads(open(config.DATASET_RGBMEAN).read())

mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()

print("[INFO] loading model...")
trainedHeadModel=load_model(TRAINED_HEAD_MODEL_PATH)

print("[INFO] predicting on test data...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size, preprocessors=[mp, itap], classes=class_num)
predictions1 = trainedHeadModel.predict_generator(testGen.generator(), steps=testGen.numImages // batch_size, max_queue_size=batch_size * 2)

(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-1: {:.2f}%".format(rank1 * 100))
(rank1, rank5) = rank5_accuracy(predictions1, testGen.db["labels"])
print("[INFO] trained head only rank-5: {:.2f}%".format(rank5 * 100))
# print(classification_report(testGen["labels"][0:testGen["labels"].shape[0]].argmax(axis=1), predictions1.argmax(axis=1), target_names=testGen["label_names"][0:testGen["label_names"].shape[0]]))
testGen.close()