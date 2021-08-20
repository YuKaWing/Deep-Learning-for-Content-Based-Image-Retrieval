import os
from shutil import copyfile
import collections

datasetPath="datasets/oxbuild_images/images"
imagesPath="datasets/oxbuild_images/images_train"
path="datasets/oxbuild_images/text_files_paths"

txtPaths=[]
images=[]
images1=[]
images2=[]
for file in os.listdir(path):
    txtPaths.append(os.path.join(path,file))

for txtPath in txtPaths:
    file = open(txtPath, "r")
    clearness = ((txtPath.split(os.path.sep)[-1]).split("_")[-1]).split(".")[0]
    label=(txtPath.split(os.path.sep)[-1]).split("_1_")[0]
    while file:
        line  = file.readline()
        if line == "":
            break
        images.append(line.split("\n")[0]+".jpg")
        images2.append(os.path.join(imagesPath, label, clearness, line.split("\n")[0]+".jpg"))
        copyfile(os.path.join(datasetPath,line.split("\n")[0]+".jpg"), os.path.join(imagesPath, label, clearness, line.split("\n")[0]+".jpg"))
    file.close() 

images1=images

for image0 in os.listdir(datasetPath):
    distractor = True
    for image1 in images:
        if image0 == image1:
            distractor = False
    if distractor == True:
        images1.append(image0)
        images2.append(os.path.join(imagesPath, "distractor", image0))
        copyfile(os.path.join(datasetPath,image0), os.path.join(imagesPath, "distractor", image0))
        
print(len(images1))

count=[0,0,0,0]
for img0 in [item for item, count in collections.Counter(images1).items() if count > 1]:
    for img1 in images2:
        if img1.split(os.path.sep)[-1] == img0:
            print(img1)
            if img1.split(os.path.sep)[-2] == "good":
                count[0]+=1
            elif  img1.split(os.path.sep)[-2] == "ok":
                count[1]+=1
            elif  img1.split(os.path.sep)[-2] == "junk":
                count[2]+=1
            elif  img1.split(os.path.sep)[-2] == "distractor":
                count[3]+=1
            os.remove(img1)
print(count)