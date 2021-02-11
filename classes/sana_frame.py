
import os
import sys
import cv2
import numpy as np
from copy import copy
from scipy.ndimage.filters import gaussian_filter

import sana_geo
from sana_color_deconvolution import StainSeparator

# TODO: this should probably be it's own file
# TODO: see where else cv2 can be used
class Frame:
    def __init__(self, img):
        self.img = img
        self.size = np.array((self.img.shape[1], self.img.shape[0]))
        if img.shape[-1] == 3:
            self.color_histo = self.histogram()

    def to_gray(self):
        if self.img.shape[-1] >= 3:
            float_img = self.img.astype(np.float64)
            self.img = np.dot(
                float_img[:, :, :3], [0.2989, 0.5870, 0.1140])[:, :, None]
            self.gray_histo = self.histogram()

    def to_rgb(self):
        if self.img.shape[-1] == 1:
            self.img = np.tile(self.img, 3)

    def round(self):
        self.img = np.rint(self.img).astype(np.uint8)

    def histogram(self):
        self.round()
        histogram = np.zeros((256, self.img.shape[-1]))
        for i in range(histogram.shape[-1]):
            histogram[:, i] = np.histogram(self.img[:, :, i],
                                           bins=256, range=(0, 255))[0]
        return histogram

    def crop(self, loc, size):
        loc = sana_geo.round(copy(loc))
        size = sana_geo.round(copy(size))
        return Frame(self.img[loc[1]:loc[1]+size[1], loc[0]:loc[0]+size[0]])

    def rescale(self, ds, size=None):
        img = np.kron(self.img[:, :, 0], np.ones((ds, ds), dtype=np.uint8))
        img = img[:, :, None]

        # NOTE: sometimes the rounding is off by a pixel
        # TODO: make sure this doesn't cause alignment issues
        if size is not None:
            img = img[:size[0], :size[1]]

        return Frame(img)

    # calculates the background color as the most common color
    #  in the grayscale space
    def get_bg_color(self):
        self.to_gray()
        histo = self.histogram()
        return np.tile(np.argmax(histo), 3)

    # apply a binary mask to the image
    def mask(self, mask, value=None):
        self.img = cv2.bitwise_and(
            self.img, self.img, mask=mask.img)[:, :, None]
        if value is not None:
            self.img[self.img == 0] = value

    # TODO: use cv2
    # TODO: define sigma in terms of microns
    def gauss_blur(self, sigma):
        self.img = gaussian_filter(self.img, sigma=sigma)
        self.blur_histo = self.histogram()

    def threshold(self, threshold, x, y):

        # prepare img arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        self.img = np.where(self.img < threshold, x, y)

    # separates the stain from the image, grays and blurs the frame
    def to_dab_gray(self, blur=0):

        # perform the stain separation
        separator = StainSeparator('H-DAB')
        _, dab, _ = separator.run(
            self.img, ret=(False, True, False), rescale=False)

        # set the new img
        self.img = dab

        # convert to grayscale
        self.to_gray()

        # blur as needed
        if blur != 0:
            self.gauss_blur(blur)

#
# end of Frame

#
# end of file
