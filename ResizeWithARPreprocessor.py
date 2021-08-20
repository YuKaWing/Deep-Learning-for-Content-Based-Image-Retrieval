import cv2

class ResizeWithARPreprocessor:
    def __init__(self, width = None, height = None, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        dim = None
        (h, w) = image.shape[:2]
    
        if self.width is None and self.height is None:
            return image
    
        if self.width is None:
            r = self.height / float(h)
            dim = (int(w * r), self.height)
    
        elif self.height is None:
            r = self.width / float(w)
            dim = (self.width, int(h * r))
    
        resized = cv2.resize(image, dim, interpolation=self.inter)
    
        return resized