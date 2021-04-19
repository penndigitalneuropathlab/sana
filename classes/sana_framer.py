
from copy import copy
import numpy as np

from sana_geo import Point

class Framer:
    def __init__(self, loader, size, step=None,
                 fpad=None, fshift=None, locs=None,
                 loc0=None, size0=None):

        # store the slide loader
        self.loader = loader
        self.converter = self.loader.converter

        # define the frame size and step
        self.size = size
        if step is None:
            self.step = copy(self.size)
        else:
            self.step = step

        # convert the dimensions to pixels and round
        if self.size.is_micron:
            self.converter.to_pixels(self.size, self.loader.lvl)
        self.converter.rescale(self.size, self.loader.lvl)
        if self.step.is_micron:
            self.converter.to_pixels(self.step, self.loader.lvl)
        self.converter.rescale(self.size, self.loader.lvl)

        self.size = np.rint(self.size).astype(np.int)
        self.step = np.rint(self.step).astype(np.int)

        # store the padding and shifting for center-alignment
        if fpad is None:
            self.fpad = Point(0, 0, False, self.loader.lvl)
            self.fshift = np.copy(self.fpad)
        else:
            self.fpad = fpad
            self.fshift = fshift

        # define the locations of the frames
        if locs is None:

            # define the origin and size of region to frame
            if size0 is None:
                size0 = self.loader.get_dim()
            if loc0 is None:
                loc0 = Point(0, 0, False, self.loader.lvl)
            if size0.is_micron:
                self.converter.to_pixels(size0, self.loader.lvl)
            self.converter.rescale(size0, self.loader.lvl)
            if loc0.is_micron:
                self.converter.to_pixels(loc0, self.loader.lvl)
            self.converter.rescale(loc0, self.loader.lvl)
            size0 = np.rint(size0).astype(np.int)
            loc0 = np.rint(loc0).astype(np.int)

            # calculate the number of frames in the region
            self.n = (size0 // self.step) + 1
            self.ds = size0 / self.n
            self.locs = [[[] for j in range(self.n[1])] \
                         for i in range(self.n[0])]
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    x, y = i * self.step[0], j * self.step[1]
                    self.locs[i][j] = \
                        Point(x, y, False, self.loader.lvl) + loc0
        else:
            self.n = len(locs)
            self.locs = locs
            for i in range(self.n):
                if self.locs[i].is_micron:
                    self.convert.to_pixels(self.locs[i], self.loader.lvl)
                self.converter.rescale(self.locs[i], self.loader.lvl)
                self.locs[i] = np.rint(self.locs[i], dtype=np.int)

    def load(self, i, j=None):
        if j is None:
            loc = self.locs[i]
        else:
            loc = self.locs[i][j]

        # load the frame, apply the shifting and padding
        return self.loader.load_frame(loc-self.fshift, self.size+self.fpad,
            pad_color=self.loader.slide_color)

    def inds(self):
        if type(self.n) is int:
            for i in range(self.n):
                yield i, None
        else:
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    yield i, j

    def frames(self):
        for i, j in self.inds():
            yield self.load(i, j)
#
# end of Framer

#
# end of file
