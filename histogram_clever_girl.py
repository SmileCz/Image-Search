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
histogram = cv2.calcHist([gray], [0], None, [256], [0, 255])

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
    # concatenate the resulting histograms for each
    # channel
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 255])
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

# let's move on to 2D histograms -- I am reducing the
# number of bins in the histogram from 256 to 32 so we
# can better visualize the results
fig = plt.figure()

# plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
histogram = cv2.calcHist([channels[1], channels[0]], [0, 1], None,
                         [32, 32], [0, 255, 0, 255])
p = ax.imshow(histogram, interpolation="nearest")
ax.set_title("2D Color Histogram for Green and Blue")
plt.colorbar(p)

# plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
histogram = cv2.calcHist([channels[1], channels[2]], [0, 1], None,
                         [32, 32], [0, 255, 0, 255])
p = ax.imshow(histogram, interpolation="nearest")
ax.set_title("2D Color Histogram for Green and Red")
plt.colorbar(p)

# plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
histogram = cv2.calcHist([channels[0], channels[2]], [0, 1], None,
                         [32, 32], [0, 255, 0, 255])
p = ax.imshow(histogram, interpolation="nearest")
ax.set_title("2D Color Histogram for Blue and Red")
plt.colorbar(p)
plt.show()

# finally, let's examine the dimensionality of one of
# the 2D histograms
print("2D histogram shape:", histogram.shape, " with", histogram.flatten().shape[0], "values")

# our 2D histogram could only take into account 2 out
# of the 3 channels in the image so now let's build a
# 3D color histogram (utilizing all channels) with 8 bins
# in each direction -- we can't plot 3D histogram, but
# the theory is exactly like that of a 2D histogram, so
# we'll just show the shape of the histrogram
histogram = cv2.calcHist([image], [0, 1, 2], None,
                         [8, 8, 8], [0, 255, 0, 255, 0, 255])
print("3D histogram shape %s, with %d values" % (histogram.shape, histogram.flatten().shape[0]))
