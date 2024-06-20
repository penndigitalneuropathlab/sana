
# installed modules
import numpy as np
from matplotlib import pyplot as plt
import tqdm

# sana modules
import sana.image
from sana.geo import Point, Polygon

class Convolver:
    """
    Class which calculates information across a series of tiles within the given Frame. This class for example generates a heatmap of the detected object density within an image
    :param logger: sana.logging.Logger
    :param frame: input sana.image.Frame
    :param tsize: (h,w) size of tiles
    :param tstep: (y,x) amount to step across the image during convolution
    :param objs: list of sana.geo.Polygon instances that were detected within the input frame
    """
    def __init__(self, logger, frame, tsize, tstep, mask=None, objs=[]):

        self.logger = logger
        self.frame = frame
        self.tsize = tsize
        self.tstep = tstep
        self.mask = mask
        self.objs = objs

        # create the tiles
        self.tiles = self.frame.to_tiles(self.tsize, self.tstep)
        self.ntiles = np.array(self.tiles.shape[:2])
        self.ds = np.array(frame.img.shape[:2]) / self.ntiles

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
        for j in tqdm.tqdm(range(self.tiles.shape[1])):
            for i in range(self.tiles.shape[0]):
                
                # TODO: debug message?

                # get the location and size of the current tile
                loc = Point(j * self.tstep[1], 
                            i * self.tstep[0], 
                            is_micron=False, level=self.frame.level)
                size = self.tsize

                # center align the tile
                loc -= size // 2

                # build the tile polygon
                if self.mask is None:
                    x = [loc[0], loc[0]+size[0], loc[0]+size[0], loc[0]]
                    y = [loc[1], loc[1], loc[1]+size[1], loc[1]+size[1]]
                    x = np.clip(x, 0, self.frame.img.shape[1])
                    y = np.clip(y, 0, self.frame.img.shape[0])
                    tile = Polygon(x, y, is_micron=False, level=self.frame.level)
                    self.tile_area = tile.get_area() # TODO: calculate this in microns!!!
                
                else:
                    tile_mask = sana.image.frame_like(self.mask, self.mask.get_tile(loc, size, pad=True))
                    c, h = tile_mask.get_caontours()
                    if len(c) != 0:
                        tile = c[0].polygon.connect()
                        tile.translate(-loc)
                        self.tile_area = tile.get_area()
                    else:
                        tile = None
                        self.tile_area = 0

                # TODO: make a anisotropic gaussian kernel (or just regular), and then mask by the tile polygon
                # ao: kernel applied to classified image
                # density: create image of 0's w/ 1's as soma_ctrs
                # area: ao / density

                # find the objects that are inside this polygon
                curr_inds = np.zeros(self.obj_ctrs.shape[0])
                if not tile is None:
                    for obj_idx in range(self.obj_ctrs.shape[0]):
                        curr_inds[obj_idx] = sana.geo.ray_tracing(
                            self.obj_ctrs[obj_idx,0],
                            self.obj_ctrs[obj_idx,1],
                            np.array(tile),
                        )
                    
                    # apply each function to the tile and the objects within the tile
                    tile_img = self.tiles[i][j]
                    if not self.mask is None:
                        tile_img[tile_mask.img[:,:,0] == 0] = 0            
                    
                    for n, func in enumerate(funcs):
                        if callable(func):
                            out[n][i][j] = func(tile_img, curr_inds)
        return out

    def ao_feature(self, img, inds):
        return np.sum(img) / self.tile_area
    
    def density_feature(self, img, inds):
        return np.sum(inds) / self.tile_area
    
    def area_feature(self, img, inds):

        # no objects found
        if np.sum(inds) == 0:
            return 0
        
        # calculate the areas of the objects within the tile
        areas = np.array([self.objs[i].get_area() for i in np.where(inds == 1)[0]])

        # only calculate the mean area of objects within a range
        # TODO: only upper cutoff?
        mu, sg = np.mean(areas), np.std(areas)
        sigma = 0.8
        cut_off = sigma * 2 # TODO: test this!
        #areas = areas[(areas > mu - cut_off) & (areas < mu + cut_off)]
        if len(areas) == 0:
            return 0

        return np.mean(areas)
    
# TODO: write Tiler
