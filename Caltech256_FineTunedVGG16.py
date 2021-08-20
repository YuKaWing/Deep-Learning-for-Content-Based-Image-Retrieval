import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import json
import os

import Config_256_ObjectCategoriess as config
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
from PatchPreprocessor import PatchPreprocessor
from RGBMeanSubtractionPreprocessor import RGBMeanSubtractionPreprocessor
#from TrainingMonitor import TrainingMonitor
from HDF5DatasetGenerator import HDF5DatasetGenerator
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImageAspectPreprocessor import ImageAspectPreprocessor
from ImageDatasetLoader import ImageDatasetLoader
from FullyConnectedHead import FullyConnectedHead

batch_size=32

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

iap = ImageAspectPreprocessor(224, 224)
itap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size, aug=aug, preprocessors=[iap, itap], classes=257)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batch_size, preprocessors=[iap, itap], classes=257)

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = FullyConnectedHead.build(baseModel, 257, 256)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False
    
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // batch_size,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // batch_size,
    epochs=25,
    max_queue_size=batch_size * 2, verbose=1)

print("[INFO] evaluating after initialization...")

for layer in baseModel.layers[15:]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning model...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // batch_size,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // batch_size,
    epochs=100,
    max_queue_size=batch_size * 2, verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
trainGen.close()
valGen.close()
