import numpy as np

a=np.array([1,2,3])
print(a)
b=list(a)
print(b)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
model = VGG16(weights="imagenet", include_top=True)
print(model.summary())