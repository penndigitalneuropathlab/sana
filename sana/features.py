
# installed modules
import numpy as np
from matplotlib import pyplot as plt

# sana modules
from sana.geo import Point, Polygon
from sana_tiler import Tiler

class Convolver:
    """
    Class which calculates information across a series of tiles within the given Frame. This class for example generates a heatmap of the detected object density within an image
    :param logger: sana.logging.Logger
    :param frame: input sana.image.Frame
    :param tsize: (h,w) size of tiles
    :param tstep: (y,x) amount to step across the image during convolution
    :param objs: list of sana.geo.Polygon instances that were detected within the input frame
    """
    def __init__(self, logger, frame, tsize, tstep, objs=[]):

        self.logger = logger
        self.frame = frame
        self.tsize = tsize
        self.tstep = tstep
        self.objs = objs

        # create the tiles
        self.tiler = Tiler(self.frame.level, self.frame.converter, self.tsize, step=self.tstep)
        self.tiler.set_frame(self.frame)
        self.tiles = self.tiler.load_tiles()

        # pre-calculate the centers of each object to speed up finding them
        # TODO: use a better metric than the mean of vertices (i.e. center of gravity)
        self.obj_ctrs = np.array([np.mean(obj, axis=0) for obj in self.objs])

    def run(self, funcs):
        """
        This function runs the convolution process, and applies the functions in the input list to each tile. These functions should be of the signature -- func(sana.image.Frame, [sana.geo.Polygon, ...])
        """

        out = np.zeros((len(funcs), self.tiles.shape[0], self.tiles.shape[1]), dtype=float)

        # sometimes we just give it an image, store it
        # TODO: is this necessary?
        for i, func in enumerate(funcs):
            if not callable(func):
                out[i, :, :] = func

        # start the convolution
        for j in range(self.tiles.shape[1]):
            for i in range(self.tiles.shape[0]):
                
                # TODO: debug message?

                # get the location and size of the current tile
                loc = Point(j * self.tiler.step[1], 
                            i * self.tiler.step[0], 
                            is_micron=False, level=self.frame.level)
                size = self.tiler.size

                # center align the tile
                loc -= size // 2

                # build the tile polygon, clipping to the image boundaries
                x = [loc[0], loc[0]+size[0], loc[0]+size[0], loc[0]]
                y = [loc[1], loc[1], loc[1]+size[1], loc[1]+size[1]]
                x = np.clip(x, 0, self.frame.img.shape[1])
                y = np.clip(y, 0, self.frame.img.shape[0])
                tile = Polygon(x, y, is_micron=False, level=self.frame.level)
                self.tile_area = tile.get_area() # TODO: calculate this in microns!!!

                # find the objects that are inside this polygon
                curr_inds = (
                    (self.obj_ctrs[:,0] > x[0]) & # left bound
                    (self.obj_ctrs[:,0] < x[1]) & # right bound
                    (self.obj_ctrs[:,1] > y[0]) & # upper bound
                    (self.obj_ctrs[:,1] < y[2])   # lower bound
                )

                # apply each function to the tile and the objects within the tile
                for n, func in enumerate(funcs):
                    if callable(func): # TODO: necessary??
                        out[n][i][j] = func(self.tiles[i][j], curr_inds)
        return out

    def density_feature(self, img, inds):
        return np.sum(inds) / self.tile_area
    
    def area_feature(self, img, inds):

        # no objects found
        if np.sum(inds) == 0:
            return 0
        
        # calculate the areas of the objects within the tile
        areas = np.array([self.objs[i].get_area() for i in range(len(inds)) if inds[i] == 1])

        # only calculate the mean area of objects within a range
        # TODO: only upper cutoff?
        mu, sg = np.mean(areas), np.std(areas)
        sigma = 0.3
        cut_off = sigma * 2 # TODO: test this!
        # areas = areas[(areas > mu - cut_off) & (areas < mu + cut_off)]
        if len(areas) == 0:
            return 0

        return np.mean(areas)
    
# TODO: write Tiler