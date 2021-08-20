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

import Config_256_ObjectCategoriess as config
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

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

batch_size=32
class_num=257
MODEL_PATH="datasets/256_ObjectCategories/output/VGG16_256_ObjectCategories.model"
TRAINED_HEAD_MODEL_PATH="datasets/256_ObjectCategories/output/VGG16_256_ObjectCategories_trained_head.model"

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")


means = json.loads(open(config.DATASET_RGBMEAN).read())

sp = ImagePreprocessor(224, 224)
pp = PatchPreprocessor(224, 224)
mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size, aug=aug, preprocessors=[pp, mp, itap], classes=class_num)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batch_size, preprocessors=[sp, mp, itap], classes=class_num)

print("[INFO] loading model...")
model = load_model(TRAINED_HEAD_MODEL_PATH)

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[15:]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.0001, decay=0.0001 / 60)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning model...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // batch_size,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // batch_size,
    epochs=60,
    max_queue_size=batch_size * 2,
    callbacks=callbacks, 
    verbose=1)

print("[INFO] serializing model...")
model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()