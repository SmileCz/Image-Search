# import the necessary packages
import imutils
import cv2


class RGBHistogram:
    def __init__(self, bins):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        histogram = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        if imutils.is_cv2():
            histogram = cv2.normalize(histogram)
        else:
            histogram = cv2.normalize(histogram,histogram)

        return histogram.flatten()
