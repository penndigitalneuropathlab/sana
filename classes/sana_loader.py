
# system packages
import os
import sys
from copy import copy

# installed packages
import openslide
import numpy as np

# custom packages
import sana_io
from sana_geo import Converter, Point
from sana_frame import Frame, get_csf_threshold

# provides an interface to initalize and load SVS files
# uses OpenSlide to do this
class Loader(openslide.OpenSlide):
    def __init__(self, fname):

        # initialize the object with the slide
        self.fname = sana_io.get_fullpath(fname)
        super().__init__(self.fname)

        # define the necessary attributes
        self.lc = self.level_count
        self.dim = self.level_dimensions
        self.ds = self.level_downsamples
        self.mpp = float(self.properties['aperio.MPP'])
        self.converter = Converter(self.mpp, self.ds)

        # pre-load the thumbnail for easy access
        self.thumbnail = self.load_thumbnail()

        # calculate the color of the slide background
        self.slide_color = copy(self.thumbnail).get_bg_color()

        # calculate the Slide/Tissue threshold
        self.csf_threshold = get_csf_threshold(copy(self.thumbnail))
    #
    # end of constructor

    # getters
    def get_dim(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        dim = self.dim[lvl]
        return Point(dim[0], dim[1], False, lvl)
    def get_ds(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        return self.ds[lvl]

    # setters
    def set_lvl(self, lvl):
        self.lvl = lvl

    # loads the entire image at the lowest resolution
    def load_thumbnail(self):
        lvl = self.lc - 1
        h, w = self.get_dim(lvl)
        loc = Point(0, 0, False, lvl)
        size = Point(h, w, False, lvl)
        return self.load_frame(loc, size, lvl)

    # loads a Frame into memory, uses a location and size coordinate
    #  image is automatically padded if loc or size exceeds the dimensions
    #  of the SVS slide
    #  -loc: top left corner of Frame (in any unit or resolution)
    #  -size: size of Frame to load (in any unit or resolution)
    #  -lvl: resolution to use, overrides the lvl attribute stored in Loader
    #  -pad_color: color to use to pad the Frame, can use slide_color or white
    def load_frame(self, loc, size, lvl=None, pad_color=None):
        if lvl is None:
            lvl = self.lvl
        if pad_color is None:
            pad_color = (255, 255, 255)

        # make sure loc and size are in current pixel resolution
        self.converter.to_pixels(loc, lvl)
        self.converter.to_pixels(size, lvl)

        # prepare variables to calculate padding
        loc = copy(loc)
        size = copy(size)
        h, w = self.get_dim(lvl)
        padx1, pady1, padx2, pady2 = 0, 0, 0, 0

        # make sure the location isn't negative
        if loc[0] < 0:
            padx1 = int(0 - loc[0])
            loc[0] = 0
            size[0] -= padx1
        if loc[1] < 0:
            pady1 = int(0 - loc[1])
            loc[1] = 0
            size[1] -= pady1

        # make sure the size isn't past the image boundary
        if loc[0] + size[0] > h:
            padx2 = int(loc[0] + size[0] - h)
            size[0] -= padx2
        if loc[1] + size[1] > w:
            pady2 = int(loc[1] + size[1] - w)
            size[1] -= pady2

        # load the region
        # NOTE: upscale the location before accessing the image
        self.converter.rescale(loc, 0)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        img = np.array(self.read_region(
            location=loc, level=lvl, size=size))[:, :, :3]

        # pad img with rows at the top/bottom with cols of img size
        pady1 = np.full_like(img, pad_color, shape=(pady1, img.shape[1], 3))
        pady2 = np.full_like(img, pad_color, shape=(pady2, img.shape[1], 3))
        img = np.concatenate((pady1, img, pady2), axis=0)

        # pad img with cols at the left/right with rows of the padded img size
        padx1 = np.full_like(img, pad_color, shape=(img.shape[0], padx1, 3))
        padx2 = np.full_like(img, pad_color, shape=(img.shape[0], padx2, 3))
        img = np.concatenate((padx1, img, padx2), axis=1)

        return Frame(img, lvl=lvl, converter=self.converter, csf_threshold=self.csf_threshold)
    #
    # end of load_frame
#
# end of Loader

#
# end of file
