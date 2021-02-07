import cv2

class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins,
            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()
        #flatten the list
        """ex:When we instantiate our RGBHistogram, we will use 8 bins per
         channel. Without flattening our histogram, the shape would be
          (8, 8, 8). But by flattening it, the shape becomes (512,)."""
