
import os
import sys
import cv2
from copy import copy
import numpy as np

import sana_geo
from sana_framer import Framer, Frame

class Tiler:
    def __init__(self, loader, tsize, tstep=None,
                 fsize=None, fstep=None):

        # store the slide loader
        self.loader = loader

        # define the tile size and step
        self.size = tsize
        if tstep is None:
            self.step = copy(self.size)
        else:
            self.step = tstep

        # convert the dimensions to pixels and round
        if self.size.is_micron:
            sana_geo.to_pixels(self.size, loader.lvl)
        else:
            sana_geo.rescale(self.size, loader.lvl)
        if self.step.is_micron:
            sana_geo.to_pixels(self.size, loader.lvl)
        else:
            sana_geo.rescale(self.size, loader.lvl)
        self.size = sana_geo.round(self.size)
        self.step = sana_geo.round(self.step)

        # define the amount to pad/shift the frames for center-alignment
        self.fpad = self.size - 1
        self.fshift = self.size // 2

        # generate the Framer object if needed
        if fsize is not None:
            self.framer = Framer(
                self.loader, fsize, fstep, fpad=self.fpad, fshift=self.fshift)

    def load_frame(self, i, j):
        return self.framer.load(i, j)

    def load_tiles(self, frame):

        # set the frame and calculate the number of tiles in the frame
        self.frame = frame
        self.n = ((self.frame.size - self.size) // self.step) + 1
        self.ds = (self.frame.size - self.fpad) / self.n

        # define the output shape
        shape = (self.n[0], self.n[1], self.size[0], self.size[1])

        # size of each element in the frame
        N = self.frame.img.itemsize

        # number of bytes between tile rows
        s0 = N * self.frame.size[0] * self.step[0]

        # number of bytes between tile cols
        s1 = N * self.step[1]

        # number of bytes between element rows
        s2 = N * self.frame.size[0]

        # number of bytes between element cols
        s3 = N * 1

        # define the stride lengths in each dimension
        strides = (s0, s1, s2, s3)

        # perform the tiling
        # NOTE: this is pretty complicated... but very fast
        #       23) in this article -> https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        self.tiles = np.lib.stride_tricks.as_strided(
            self.frame.img, shape=shape, strides=strides)
        return self.tiles
#
# end of Tiler

#
# end of file
