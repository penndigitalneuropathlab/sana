
import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from copy import copy
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import sana_io
import sana_geo
from sana_color_deconvolution import StainSeparator

# TODO: see where else cv2 can be used
class Frame:
    def __init__(self, img):
        self.img = img
        self.size = np.array((self.img.shape[1], self.img.shape[0]))
        if img.shape[-1] == 3:
            self.color_histo = self.histogram()

    def copy(self):
        return Frame(np.copy(self.img))

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

    # NOTE: reshape=False might cause alignment issue down the line
    def rotate(self, angle):
        img = ndimage.rotate(self.img, angle, reshape=False, mode='nearest')
        return Frame(img)

    def save(self, fname):
        sana_io.create_directory(fname)
        if self.img.ndim == 3 and self.img.shape[2] == 1:
            im = self.img[:, :, 0]
        else:
            im = self.img
        im = Image.fromarray(im)
        im.save(fname)

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
#
# end of Frame

#
# end of file
