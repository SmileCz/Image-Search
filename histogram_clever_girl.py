# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("image", image)

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()

# grab the image channels, initialize the tuple of colors,
# the figure and flattened feature vector
channels = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
for (channel, color) in zip(channels, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histrograms for each
    # channel
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    features.extend(histogram)

    # plot histogram
    plt.plot(histogram, color=color)
    plt.xlim([0, 256])

plt.show()
# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print("flattened features vector size: ", np.array(features).flatten().shape)

cv2.waitKey(0)
