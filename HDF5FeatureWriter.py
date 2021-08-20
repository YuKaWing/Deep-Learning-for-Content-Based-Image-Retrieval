import h5py
import os

class HDF5FeatureWriter:
    def __init__(self, dims, outputPath, dataKey="feature", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)
            
        self.db = h5py.File(outputPath, "w")
        self.feature = self.db.create_dataset(dataKey, dims, dtype="float")
        dt = h5py.special_dtype(vlen=str)
        self.imageID = self.db.create_dataset("imageID", (dims[0],), dtype=dt)
        self.bufSize = bufSize
        self.buffer = {"feature": [], "imageID": []}
        self.idx = 0
            
    def add(self, rows , imageID):
        self.buffer["feature"].extend(rows)
        self.buffer["imageID"].extend(imageID)
        
        if len(self.buffer["feature"]) >= self.bufSize:
            self.flush()
            
    def flush(self):
        i = self.idx + len(self.buffer["feature"])
        self.feature[self.idx:i] = self.buffer["feature"]
        self.imageID[self.idx:i] = self.buffer["imageID"]
        self.idx = i
        self.buffer = {"feature": [], "imageID": []}
        
    def close(self):
        if len(self.buffer["feature"]) > 0:
            self.flush()
        self.db.close()