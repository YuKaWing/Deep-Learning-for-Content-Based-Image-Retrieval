import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import json
import os
import numpy as np

import Config_256_ObjectCategoriess_1 as config
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
from PatchPreprocessor import PatchPreprocessor
from ImageMeanSubtractionPreprocessor import ImageMeanSubtractionPreprocessor
from TrainingMonitor import TrainingMonitor
from HDF5DatasetGenerator import HDF5DatasetGenerator
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImageAspectPreprocessor import ImageAspectPreprocessor
from ImageDatasetLoader import ImageDatasetLoader
from FullyConnectedHead import FullyConnectedHead

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import load_model

batch_size=32
class_num=config.NUM_CLASSES
MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "ResNet50_256_ObjectCategories.hdf5"])
TRAINED_HEAD_MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "ResNet50_256_ObjectCategories_trained_head.hdf5"])

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=False, fill_mode="nearest")

imageMean=np.load(config.DATASET_IMAGEMEAN)

mp = ImageMeanSubtractionPreprocessor(imageMean)
itap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size, aug=aug, preprocessors=[mp, itap], classes=class_num)
valGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size, preprocessors=[mp, itap], classes=class_num)

model = load_model(TRAINED_HEAD_MODEL_PATH)
print(model.summary())

for layer in model.layers:
    layer.trainable = True

print("[INFO] re-compiling model...")
path = os.path.sep.join([config.OUTPUT_PATH, "ResNet50_256_ObjectCategories_train_model.png"])
callbacks = [TrainingMonitor(path)]
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning model...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // batch_size + 1,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // batch_size + 1,
    epochs=40,
    max_queue_size=batch_size * 2,
    callbacks=callbacks, 
    verbose=1)

print("[INFO] serializing model...")
model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()