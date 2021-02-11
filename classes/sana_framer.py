
import os
import sys
from copy import copy
import numpy as np

import sana_geo
from sana_frame import Frame

class Framer:
    def __init__(self, loader, size, step=None,
                 fpad=None, fshift=None, locs=None):

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

        # store the padding and shifting for center-alignment
        if fpad is None:
            self.fpad = sana_geo.Point(0, 0, self.loader.mpp, self.loader.ds,
                                       is_micron=False, lvl=self.loader.lvl)
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
            self.locs = locs
            for i in range(self.n):
                sana_geo.to_pixels(self.locs[i], self.loader.lvl)
            self.i = 0

    def load(self, i, j=None):
        if j is None:
            loc = self.locs[i]
        else:
            loc = self.locs[i][j]

        # load the frame, apply the shifting and padding
        return self.loader.load_frame(loc-self.fshift, self.size+self.fpad,
            pad_color=self.loader.slide_color)
#
# end of Framer

#
# end of file
