
import os
import sys
import cv2
from copy import copy
from scipy.ndimage.filters import gaussian_filter
import numpy as np

import sana_geo

class Framer:
    def __init__(self, loader, size, step=None, fpad=None, fshift=None, locs=None):

        # store the slide loader
        self.loader = loader

        # define the frame size and step
        self.size = size
        if step is None:
            self.step = copy(self.size)
        else:
            self.step = step

        # convert the dimensions to pixels and round
        if self.size.is_micron:
            sana_geo.to_pixels(self.size, loader.lvl)
        else:
            sana_geo.rescale(self.size, loader.lvl)
        if self.step.is_micron:
            sana_geo.to_pixels(self.step, loader.lvl)
        else:
            sana_geo.rescale(self.size, loader.lvl)
        self.size = sana_geo.round(self.size)
        self.step = sana_geo.round(self.step)

        # store the padding and shifting for center-alignment
        if fpad is None:
            self.fpad = Point(0, 0, self.loader.mpp, self.loader.ds, is_micron=False, lvl=self.loader.lvl)
            self.fshift = np.copy(self.fpad)
        else:
            self.fpad = fpad
            self.fshift = fshift

        # define the locations of the frames
        if locs is None:

            # calculate the number of frames in the slide
            self.n = (self.loader.get_dim() // self.step) + 1
            self.ds = self.loader.get_dim() / self.n

            self.locs = [[[] for j in range(self.n[1])] \
                         for i in range(self.n[0])]
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    x, y = i * self.step[0], j * self.step[1]
                    self.locs[i][j] = \
                        sana_geo.Point(x, y, self.loader.mpp,
                                       self.loader.ds, is_micron=False,
                                       lvl=self.loader.lvl)
        else:
            self.n = len(locs)
            self.ds = self.loader.get_dim() / self.n
            self.locs = locs

    def load(self, i, j):

        # load the frame, apply the shifting and padding
        return self.loader.load_frame(
            self.locs[i][j]-self.fshift, self.size+self.fpad,
            pad_color=self.loader.slide_color)

# TODO: this should probably be it's own file
# TODO: see where else cv2 can be used
class Frame:
    def __init__(self, img):
        self.img = img
        self.size = np.array((img.shape[1], img.shape[0]))

    def to_gray(self):
        if self.img.shape[-1] >= 3:
            float_img = self.img.astype(np.float64)
            self.img = np.dot(
                float_img[:, :, :3], [0.2989, 0.5870, 0.1140])

    def to_rgb(self):
        if self.img.shape[-1] == 1:
            self.img = np.tile(self.img, 3)

    def round(self):
        self.img = np.rint(self.img).astype(np.uint8)

    def histogram(self):
        self.round()
        return np.histogram(self.img, bins=256, range=(0, 255))[0]

    # calculates the background color as the most common color
    #  in the grayscale space
    def get_bg_color(self):
        self.to_gray()
        histo = self.histogram()
        return np.tile(np.argmax(histo), 3)

    # apply a binary mask to the image
    def mask(self, mask):
        self.img = cv2.bitwise_and(self.img, self.img, mask=mask.img)

    # TODO: use cv2
    # TODO: define sigma in terms of microns
    def gauss_blur(self, sigma):
        self.img = gaussian_filter(self.img, sigma=sigma)

    def threshold(self, threshold, x, y):

        # prepare img arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        self.img = np.where(self.img < threshold, x, y)
#
# end of Frame

#
# end of file
