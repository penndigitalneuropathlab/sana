
from copy import copy
import numpy as np

from sana_framer import Framer

class Tiler:
    def __init__(self, lvl, converter, tsize, tstep=None,
                 fsize=None, fstep=None, loader=None):
        self.lvl = lvl
        self.converter = converter

        # define the tile size and step size
        self.size = tsize
        if tstep is None:
            self.step = copy(self.size)
        else:
            self.step = tstep

        # convert the dimensions to pixels and round
        if self.size.is_micron:
            self.converter.to_pixels(self.size, self.lvl)
        self.converter.rescale(self.size, self.lvl)
        if self.step.is_micron:
            self.converter.to_pixels(self.step, self.lvl)
        self.converter.rescale(self.step, self.lvl)

        self.size = np.rint(self.size).astype(np.int)
        self.step = np.rint(self.step).astype(np.int)

        # define the amount to pad/shift the frames for center-alignment
        self.fpad = self.size - 1
        self.fshift = self.size // 2

        # generate the Framer object if needed
        if fsize is not None:
            self.framer = Framer(
                loader, fsize, fstep, fpad=self.fpad, fshift=self.fshift)

    def load_frame(self, i, j):
        return self.framer.load(i, j)

    def set_frame(self, frame, pad=False):
        if pad:
            frame = frame.pad(self.fpad)

        # set the frame and calculate the number of tiles in the frame
        self.frame = frame
        self.frame.img = self.frame.img[:, :, 0]
        self.n = ((self.frame.size() - self.size) // self.step) + 1
        self.ds = (self.frame.size() - self.fpad) / self.n

    def get_tile_bounds(self, frame=None, pad=False):
        if not frame is None:
            self.set_frame(frame, pad)

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                x0 = i * self.step[0]
                y0 = j * self.step[1]
                x1 = x0 + self.size[0]
                y1 = y0 + self.size[1]
                yield i, j, x0, y0, x1, y1

    def load_tiles(self, frame=None, pad=False):
        if not frame is None:
            self.set_frame(frame, pad)

        # define the output shape
        shape = (self.n[1], self.n[0], self.size[1], self.size[0])

        # size of each element in the frame
        N = self.frame.img.itemsize

        # number of bytes between tile rows
        s0 = N * self.frame.size()[0] * self.step[1]

        # number of bytes between tile cols
        s1 = N * self.step[0]

        # number of bytes between element rows
        s2 = N * self.frame.size()[0]

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
