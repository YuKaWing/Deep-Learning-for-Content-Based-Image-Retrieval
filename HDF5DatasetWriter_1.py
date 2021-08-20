import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)
            
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        dt = h5py.special_dtype(vlen=str)
        self.imageID = self.db.create_dataset("imageID", (dims[0],), dtype=dt)
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": [], "imageID": []}
        self.idx = 0
            
    def add(self, rows, labels, imageID):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        self.buffer["imageID"].extend(imageID)
        
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
            
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.imageID[self.idx:i] = self.buffer["imageID"]
        self.idx = i
        self.buffer = {"data": [], "labels": [], "imageID": []}
        
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels
        
    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()