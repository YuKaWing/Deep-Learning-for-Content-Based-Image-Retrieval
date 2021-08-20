import h5py
import numpy as np
import Config_oxbuild_images_1 as config
import cv2
from ImageMeanSubtractionPreprocessor import ImageMeanSubtractionPreprocessor
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from ImagePreprocessor import ImagePreprocessor
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from Searcher import Searcher
import progressbar

modelPath = os.path.sep.join([config.OUTPUT_PATH, "ResNet50_oxbuild_images.hdf5"])
imageMean=np.load(config.DATASET_IMAGEMEAN)

model = load_model(modelPath)
x = model.layers[-2].output
model = Model(inputs = model.input, outputs = x)

ip = ImagePreprocessor(224, 224)
mp = ImageMeanSubtractionPreprocessor(imageMean)
itap = ImageToArrayPreprocessor()

txtsPath="datasets/oxbuild_images/gt_files_170407"

queries=[]
for txt in os.listdir(txtsPath):
    if txt.split('_')[-1] == "query.txt":
        queries.append(txt.split("query.txt")[0])

apList=[]

widgets = ["Evaluating Search Engine: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queries),
widgets=widgets).start()
for (i_pbar, query) in enumerate(queries):
    file = open(os.path.join(txtsPath, query+"query.txt"), "r")
    queryImageID  = (file.readline().split(' ')[0]).split('oxc1_')[-1]
    file.close()
    
    posImages=[]
    nullImages=[]
    
    file = open(os.path.join(txtsPath, query+"good.txt"), "r")
    while file:
        line  = file.readline()
        if line == "":
            break
        posImages.append(line.split('\n')[0])
    file.close()
    
    file = open(os.path.join(txtsPath, query+"ok.txt"), "r")
    while file:
        line  = file.readline()
        if line == "":
            break
        posImages.append(line.split('\n')[0])
    file.close() 
    
    file = open(os.path.join(txtsPath, query+"junk.txt"), "r")
    while file:
        line  = file.readline()
        if line == "":
            break
        nullImages.append(line.split('\n')[0])
    file.close() 
    nullImages.append(queryImageID)
    
    batchImages = []
    
    queryImage = cv2.imread(os.path.join(config.IMAGES_PATH, queryImageID+".jpg"))
    queryImage = queryImage.astype('Float64')
    queryImage = ip.preprocess(queryImage)
    queryImage = mp.preprocess(queryImage)
    queryImage = itap.preprocess(queryImage)
    queryImage = np.expand_dims(queryImage, axis=0)
    batchImages.append(queryImage)
    
    batchImages = np.vstack(batchImages)
    queryFeature = model.predict(batchImages, batch_size=20)
    queryFeature = queryFeature.reshape((queryFeature.shape[0], 256))
    
    searcher = Searcher(os.path.sep.join([config.HDF5_PATH, "features_105K_ResNet50.hdf5"]))
    
    results = searcher.search_h5py(queryFeature[0], limit=-1)
    
    ranked_list = [(result[1].split('\\')[-1]).split(".")[0] for result in results]
    
    # cv2.imshow(query, cv2.imread(os.path.join(config.IMAGES_PATH, queryImageID+".jpg")))
    # cv2.waitKey(0)

    
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
      
    intersect_size = 0;
    i = 0
    j = 0
    while i<len(ranked_list):
        null=False
        pos=False
        for x in nullImages:
            if ranked_list[i] == x:
                null = True
        for x in posImages:
            if ranked_list[i] == x:
                pos = True
        if null:
            i+=1
            continue
        if pos: 
            intersect_size+=1
    
        recall = intersect_size / float(len(posImages))
        precision = intersect_size / (j + 1.0)
    
        ap += (recall - old_recall)*((old_precision + precision)/2.0)
    
        old_recall = recall
        old_precision = precision
        
        # if j<5:
        #     cv2.imshow(query, cv2.imread(os.path.join(config.IMAGES_PATH, ranked_list[i]+".jpg")))
        #     cv2.waitKey(0)
        j+=1
        i+=1
    
    apList.append(ap)
    
    cv2.destroyAllWindows()
    pbar.update(i_pbar)

pbar.finish()
    
sum_ap = 0
for ap in apList:
    sum_ap = sum_ap + ap
mean_ap = sum_ap / len(apList)

print(mean_ap)