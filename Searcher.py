import numpy as np
import csv
import h5py

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath
        
    def search_csv(self, queryFeatures, limit = 10):
        results = {}

        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                results[row[0]] = d
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])

        return results[:limit]
    
    def search_h5py(self, queryFeatures, limit = 10):
        results = {}
        
        with h5py.File(self.indexPath) as f:
            features = f["feature"][:]
            imageIDs = f["imageID"][:]
            for (i,(imageID, feature)) in enumerate(list(zip(imageIDs,features))):
                d = self.chi2_distance(feature, queryFeatures)
                results[imageID] = d
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])

        return results[:limit]
    
    def chi2_distance(self, vecA, vecB, eps = 1e-10):

        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(vecA, vecB)])

        return d