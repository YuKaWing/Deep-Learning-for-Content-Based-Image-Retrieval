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
from TrainingMonitor1 import TrainingMonitor1
from HDF5DatasetGenerator import HDF5DatasetGenerator
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImageAspectPreprocessor import ImageAspectPreprocessor
from ImageDatasetLoader import ImageDatasetLoader
from FullyConnectedHead import FullyConnectedHead

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

batch_size=32
class_num=config.NUM_CLASSES
MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "VGG16_256_ObjectCategories"])
TRAINED_HEAD_MODEL_PATH=os.path.sep.join([config.OUTPUT_PATH, "VGG16_256_ObjectCategories_trained_head"])

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=False, fill_mode="nearest")

means = json.loads(open(config.DATASET_RGBMEAN).read())

mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size, aug=aug, preprocessors=[mp, itap], classes=class_num)
valGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size, preprocessors=[mp, itap], classes=class_num)

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(4096, activation="relu")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(class_num, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
print(model.summary())

# for layer in baseModel.layers:
#     layer.trainable = False
    
print("[INFO] compiling model...")
path = os.path.sep.join([config.OUTPUT_PATH, "VGG16_256_ObjectCategories_train_head.png"])
callbacks = [TrainingMonitor1(path)]
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // batch_size,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // batch_size,
    epochs=30,
    max_queue_size=batch_size * 2,
    callbacks=callbacks,
    verbose=1)

print("[INFO] serializing head-trained model...")
model.save(TRAINED_HEAD_MODEL_PATH, overwrite=True)

# for layer in baseModel.layers[15:]:
#     layer.trainable = True

# print("[INFO] re-compiling model...")
# path = os.path.sep.join([config.OUTPUT_PATH, "VGG16_256_ObjectCategories_train_model.png"])
# callbacks = [TrainingMonitor1(path)]
# opt = SGD(lr=0.0001, decay=0.0001 / 60)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# print("[INFO] fine-tuning model...")
# model.fit_generator(
#     trainGen.generator(),
#     steps_per_epoch=trainGen.numImages // batch_size,
#     validation_data=valGen.generator(),
#     validation_steps=valGen.numImages // batch_size,
#     epochs=60,
#     max_queue_size=batch_size * 2,
#     callbacks=callbacks, 
#     verbose=1)

# print("[INFO] serializing model...")
# model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()