import os
import cv2 as cv
import numpy as np

# Set Your Filepath to Your Files Folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
# Create Image and Show it
brain = cv.imread('np_brain_image.jpg', cv.IMREAD_UNCHANGED)
cv.imshow('Brain Scan Normal', brain)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
# Greyscale the Image
brain_gray = cv.cvtColor(brain, cv.COLOR_GRAY2BGRA)
cv.imshow('Brain Scan Grayscale', brain_gray)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
# Contrast Stretching with OpenCV Equalizer
brain_eq = cv.equalizeHist(brain)
cv.imshow('Brain Scan Contrast Stretch', brain_eq)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
# Contrast Stretch using Numpy
hist, bins = np.histogram(brain.flatten(), 256, [0, 256])
# print(hist)

cdf = hist.cumsum()
# Cumaltive Summation of the pixel regions bins
# print(cdf)

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = ((cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())) * 255
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
# Color Stretch for the pixel regions bins
# print(cdf)

brain_cs = cdf[brain]
cv.imshow('Brain Scan Contrast Stretch', brain_cs)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
# Gaussian Threshold Function
brain_thresh = cv.adaptiveThreshold(brain_cs, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 1)
cv.imshow('Brain Scan Threshold Gaussian', brain_thresh)
cv.imshow('Brain Scan Contrast Stretch', brain_cs)
cv.imshow('Brain Scan Normal', brain)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
# Global Threshold Function
thresh, brain_globe = cv.threshold(brain_cs, 125, 255, cv.THRESH_BINARY)
cv.imshow('Brain Scan Threshold Global', brain_globe)
cv.imshow('Brain Scan Contrast Stretch', brain_cs)
cv.imshow('Brain Scan Normal', brain)
cv.waitKey(0)
cv.destroyAllWindows()