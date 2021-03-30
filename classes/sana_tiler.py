
from copy import copy
import numpy as np

from sana_framer import Framer, Frame

class Tiler:
    def __init__(self, loader, tsize, tstep=None,
                 fsize=None, fstep=None):

        # store the slide loader
        self.loader = loader
        self.converter = self.loader.converter

        # define the tile size and step size
        self.size = tsize
        if tstep is None:
            self.step = copy(self.size)
        else:
            self.step = tstep

        # convert the dimensions to pixels and round
        if self.size.is_micron:
            self.converter.to_pixels(self.size, lself.oader.lvl)
        self.converter.rescale(self.size, self.loader.lvl)
        if self.step.is_micron:
            self.converter.to_pixels(self.step, self.loader.lvl)
        self.converter.rescale(self.step, self.loader.lvl)

        self.size = np.rint(self.size, dtype=np.int)
        self.step = np.rint(self.step, dtype=np.int)

        # define the amount to pad/shift the frames for center-alignment
        self.fpad = self.size - 1
        self.fshift = self.size // 2

        # generate the Framer object if needed
        if fsize is not None:
            self.framer = Framer(
                self.loader, fsize, fstep, fpad=self.fpad, fshift=self.fshift)

    def load_frame(self, i, j):
        return self.framer.load(i, j)

    def load_tiles(self, frame, pad=False):
        if pad:
            frame = frame.pad(self.fpad)

        # set the frame and calculate the number of tiles in the frame
        self.frame = frame
        self.frame.img = self.frame.img[:, :, 0]
        self.n = ((self.frame.size - self.size) // self.step) + 1
        self.ds = (self.frame.size - self.fpad) / self.n

        # define the output shape
        shape = (self.n[1], self.n[0], self.size[1], self.size[0])

        # size of each element in the frame
        N = self.frame.img.itemsize

        # number of bytes between tile rows
        s0 = N * self.frame.size[0] * self.step[1]

        # number of bytes between tile cols
        s1 = N * self.step[0]

        # number of bytes between element rows
        s2 = N * self.frame.size[0]

        # number of bytes between element cols
        s3 = N * 1

        # define the stride lengths in each dimension
        strides = (s0, s1, s2, s3)

        # perform the tiling
        # NOTE: this is pretty complicated... but very fast
        #       23) in this article -> https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        return np.lib.stride_tricks.as_strided(
            self.frame.img, shape=shape, strides=strides)
#
# end of Tiler

#
# end of file
