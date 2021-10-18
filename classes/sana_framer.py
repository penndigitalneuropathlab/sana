
# system packages
from copy import copy

# installed packages
import numpy as np

# custom packages
from sana_geo import Point

# loads a series of Frames into memory from a slide image
# use locs argument to define a list of locations to load at
# use roi_loc and roi_size to specify a larger region to load
#  -loader: Loader object to load the data
#  -size: size of Frame to load
#  -step: distance between Frames, uses size if not given
#  -locs: list of Points used as origins for Frames
#  -roi_loc: Point for origin location of large ROI
#  -roi_size: Point for size of large ROI
class Framer:
    def __init__(self, loader, size, step=None, locs=None,
                 roi_loc=None, roi_size=None,
                 fpad=None, fshift=None):

        # store the slide loader
        self.loader = loader
        self.converter = self.loader.converter
        self.lvl = self.loader.lvl
        
        # set the frame size, convert to pixels, and round
        self.size = size
        self.converter.to_pixels(self.size, self.lvl)
        self.size = self.converter.to_int(self.size)

        # set the frame step, convert to pixels, and round
        if step is None:
            self.step = copy(self.size)
        else:
            self.step = step
        self.converter.to_pixels(self.step, self.lvl)
        self.step = self.converter.to_int(self.step)

        # set the locations of the frames to be loaded
        if not locs is None:
            self.n = len(locs)
            self.locs = locs
            for i in range(self.n):
                self.converter.to_pixels(self.locs[i], self.loader.lvl)
                self.locs[i] = self.converter.to_int(self.locs[i])
        else:

            # set size of ROI, convert to pixels, and round
            if roi_size is None:
                roi_size = self.loader.get_dim()
            self.converter.to_pixels(roi_size, self.lvl)
            roi_size = self.converter.to_int(roi_size)

            # set location of ROI, convert to pixels, and round
            if roi_loc is None:
                roi_loc = Point(0, 0, False, self.loader.lvl)
            self.converter.to_pixels(roi_loc, self.lvl)
            roi_loc = self.converter.to_int(roi_loc)

            # calculate the number of frames in the ROI
            self.n = (roi_size // self.step) + 1
            self.ds = roi_size / self.n

            # calculate the frame locations within the ROI
            self.locs = [[[] for j in range(self.n[1])] \
                         for i in range(self.n[0])]
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    x, y = i * self.step[0], j * self.step[1]
                    self.locs[i][j] = \
                        Point(x, y, False, self.loader.lvl) + roi_loc
        #
        # end of locs setting

        # set the frame pad amount
        if fpad is None:
            fpad = Point(0, 0, False, self.loader.lvl)
        self.converter.to_pixels(fpad, self.lvl)
        self.fpad = self.converter.to_int(fpad)

        # set the frame shift amount
        if fshift is None:
            fshift = Point(0, 0, False, self.loader.lvl)
        self.converter.to_pixels(fshift, self.lvl)
        self.fshift = self.converter.to_int(fshift)
    #
    # end of constructor

    # loads a frame into memory
    # can be used to access manually set locs (i)
    # or is used to access the frames in an ROI (i, j)
    def load(self, i, j=None):
        if j is None:
            loc = self.locs[i]
        else:
            loc = self.locs[i][j]

        # load the frame, apply the shifting and padding
        return self.loader.load_frame(loc-self.fshift, self.size+self.fpad,
            pad_color=self.loader.slide_color)
    #
    # end of load

    # generates the list of indices used to load the frames
    def inds(self):
        if type(self.n) is int:
            for i in range(self.n):
                yield i, None
        else:
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    yield i, j

    # generator for frame loading
    def frames(self):
        for i, j in self.inds():
            yield self.load(i, j)
#
# end of Framer

#
# end of file
