import cv2

class ImageMeanSubtractionPreprocessor:
    def __init__(self, mean):
        self.mean = mean
        
    def preprocess(self, image):
        image = cv2.subtract(image,self.mean)
        return image
