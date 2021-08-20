import h5py
import numpy as np
import Config_oxbuild_images_2 as config
import cv2
from RGBMeanSubtractionPreprocessor import RGBMeanSubtractionPreprocessor
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from Searcher import Searcher
import progressbar
from ResizeWithARPreprocessor import ResizeWithARPreprocessor
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, type=str, help="path of query image")
ap.add_argument("-n", "--number", required=False, type=int, default=6, help="number of query result")
ap.add_argument("-d", "--height", required=False, type=int, default=600, help="height of displayed image")
args = vars(ap.parse_args())

IMAGES_PATH="datasets\\oxbuild_images\\images_search"

modelPath = os.path.sep.join([config.OUTPUT_PATH, "VGG16_oxbuild_images.hdf5"])
means = json.loads(open(config.DATASET_RGBMEAN).read())

model = load_model(modelPath)
x = model.layers[-2].output
model = Model(inputs = model.input, outputs = x)

ip = ImagePreprocessor(224, 224)
mp = RGBMeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
itap = ImageToArrayPreprocessor()

txtsPath="datasets/oxbuild_images/gt_files_170407"

queries=[args["query"]]

widgets = ["Searching: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queries),
widgets=widgets).start()

rwarp = ResizeWithARPreprocessor(height=args["height"])
for (i_pbar, query) in enumerate(queries):
    queryImageID  = query
    
    posImages=[]
    nullImages=[]
    
    nullImages.append(queryImageID)
    
    batchImages = []
    
    queryImage = cv2.imread(queryImageID)
    queryImage = queryImage.astype('Float64')
    queryImage = ip.preprocess(queryImage)
    queryImage = mp.preprocess(queryImage)
    queryImage = itap.preprocess(queryImage)
    queryImage = np.expand_dims(queryImage, axis=0)
    batchImages.append(queryImage)
    
    batchImages = np.vstack(batchImages)
    queryFeature = model.predict(batchImages, batch_size=20)
    queryFeature = queryFeature.reshape((queryFeature.shape[0], 256))
    
    searcher = Searcher(os.path.sep.join([config.HDF5_PATH, "features_VGG16.hdf5"]))
    
    results = searcher.search_h5py(queryFeature[0], limit=-1)
    
    ranked_list = [(result[1].split('\\')[-1]).split(".")[0] for result in results]
    
    # cv2.imshow(query, cv2.imread(os.path.join(config.IMAGES_PATH, queryImageID+".jpg")))
    # cv2.waitKey(0)

    intersect_size = 0;
    i = 0
    j = 0
    while i<len(ranked_list):
        if j<args["number"]:
            print(os.path.join(IMAGES_PATH, ranked_list[j]+".jpg"))
            img = cv2.imread(os.path.join(IMAGES_PATH, ranked_list[j]+".jpg"))
            img = rwarp.preprocess(img)
            cv2.imshow(os.path.join(IMAGES_PATH, ranked_list[j]+".jpg"), img)
            cv2.waitKey(0)
        else:
            break
        j+=1
    
    cv2.destroyAllWindows()
    pbar.update(i_pbar)

pbar.finish()
