
import os
import sys
import openslide

import sana_io

class Loader(openslide.OpenSlide):
    def __init__(self, fname):

        # initialize the object with the slide
        self.fname = sana_io.get_fullpath(fname)
        super(self.fname)

        # define the necessary attributes
        self.lc = self.level_count
        self.dim = self.level_dimensions
        self.ds = self.level_downsamples
        self.mpp = float(self.properties['aperio.MPP'])

        # pre-load the thumbnail for easy access
        self.thumbnail = self.load_thumbnail()

        # calculate the color of the slide background
        self.slide_color = self.thumbnail.get_bg_color()

#
# end of file
