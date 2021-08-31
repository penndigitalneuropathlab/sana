
# system packages
from copy import copy

# installed packages
import numpy as np

# custom packages
from sana_framer import Framer

# custom Exceptions
ERR = "---> %s <---"
ERR_FRAMER = ERR % ("Tiler doesn't have a Framer, must provide fsize and fstep")
ERR_FRAME = ERR % ("Frame must be a single channel image (m, n, 1)")
class TilerException(Exception):
    def __init__(self, message):
        self.message = message

# creates a set of tiles from a Frame with a given size and step length
# if Frames are being set manually, ignore the Framer portion of this class
# the tiles are center-aligned on the frame, some Frame padding occurs
#  -lvl: pixel resolution to use
#  -converter: Converter to handle unit conversion
#  -tsize: Point, defines the size of each tile
#  -tstep: Point, defines the distance between each tile, tsize if not given
#  -fsize: Point, size of Frame to load with Framer
#  -loader: Loader object for Framer
#  -roi_loc: Point, location of ROI for the Framer
#  -roi_size: Point, size of ROI for the Framer
class Tiler:
    def __init__(self, lvl, converter, size, step=None,
                 fsize=None, loader=None, roi_loc=None, roi_size=None):
        self.lvl = lvl
        self.converter = converter

        # set the tile size, convert to pixels, and round
        self.size = size
        self.converter.to_pixels(self.size, self.lvl)
        self.size = self.converter.to_int(self.size)

        # set the tile step, convert to pixels, and round
        if step is None:
            self.step = copy(self.size)
        else:
            self.step = step
        self.converter.to_pixels(self.step, self.lvl)
        self.step = self.converter.to_int(self.step)

        # calculate the amount to pad the frame for center alignment
        self.fpad = self.size - 1
        self.fshift = self.size // 2

        # create the Framer if needed
        if not fsize is None:
            self.framer = Framer(loader, fsize,
                                 roi_loc=roi_loc, roi_size=roi_size,
                                 fpad=self.fpad, fshift=self.fshift)
    #
    # end of constructor

    # loads a frame into memory using the Framer
    def load_frame(self, i, j):
        if not hasattr(self, 'framer'):
            raise TilerException(ERR_FRAMER)
        return self.framer.load(i, j)
    #
    # end of load_frame

    # sets the current Frame to be used by the Tiler
    def set_frame(self, frame):
        if frame.img.shape[2] != 1:
            raise TilerException(ERR_FRAME)
        self.frame = frame

        # pad the frame for center-alignment
        self.frame.pad(self.fpad)

        # TODO: check if this is necessary? might mess with the stride_tricks
        self.frame.img = self.frame.img[:, :, 0]

        # calculate the number of tiles in the frame and the downsample factor
        self.n = ((self.frame.size() - self.size) // self.step) + 1
        self.ds = (self.frame.size() - self.fpad) / self.n
    #
    # end of set_frame

    # generates a series of tiles from a Frame
    # resulting array will be the shape: (ntilesx, ntilesy, sizex, sizey)
    def load_tiles(self, frame=None):
        if not frame is None:
            self.set_frame(frame)

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
    # end of load_tiles
#
# end of Tiler

#
# end of file
