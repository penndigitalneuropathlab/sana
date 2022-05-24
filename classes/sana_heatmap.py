
# installed modules
import numpy as np

# custom modules
from sana_tiler import Tiler
from sana_geo import Point, Polygon

# TODO: better name, this is kinda a HeatmapGenerator
class Heatmap:
    def __init__(self, frame, objs, tsize, tstep, debug=False):

        self.frame = frame.copy()
        self.gray = frame.copy()
        self.gray.to_gray()
        self.objs = objs
        self.tsize = tsize
        self.tstep = tstep
        self.debug = debug
        self.lvl = self.frame.lvl
        self.converter = self.frame.converter

        self.centers = self.get_centers()
        
        self.tiler = Tiler(self.lvl, self.converter, self.tsize, self.tstep)
        self.tiler.set_frame(self.frame)
        
        self.tiles = self.tiler.load_tiles()
        self.tile_area = self.tiles[0][0].size # TODO: this is in pixels, is that right?

    # TODO: is this correct?
    def get_centers(self):
        return np.array([np.mean(obj, axis=0) for obj in self.objs])
        
    def run(self, funcs):

        feat = np.zeros((len(funcs), self.tiles.shape[0], self.tiles.shape[1]), dtype=float)
        for j in range(self.tiles.shape[1]):
            for i in range(self.tiles.shape[0]):

                # status messaging
                if self.debug:
                    s = '%d/%d' % (i+j*self.tiles.shape[0], (self.tiles.shape[0]*self.tiles.shape[1]))
                    print(s+'\b'*len(s),end="",flush=True)
                
                # get the current tile's loc and size
                loc = Point(
                    j * self.tiler.step[1],
                    i * self.tiler.step[0],
                    False, self.lvl)
                size = self.tiler.size
                
                # center align
                loc -= size // 2

                # build the tile polygon
                x = [loc[0], loc[0]+size[0], loc[0]+size[0], loc[0]]
                y = [loc[1], loc[1], loc[1]+size[1], loc[1]+size[1]]
                x = np.clip(x, 0, self.frame.img.shape[1])
                y = np.clip(y, 0, self.frame.img.shape[0])
                tile = Polygon(x, y, False, self.lvl)
                
                # get the objects that are in the tile
                # TODO: add option for other checks, like size for example
                inds = (
                    (self.centers[:,0] > x[0]) & (self.centers[:,0] < x[1]) & \
                    (self.centers[:,1] > y[0]) & (self.centers[:,1] < y[2])
                )

                # calculate and store the feature
                for n, feat_func in enumerate(funcs):
                    feat[n][i][j] = feat_func(inds)
            #
            # end of i
        #
        # end of j

        return feat
    #
    # end of run

    # TODO: check the math
    def density(self, inds):
        return 1000 * np.sum(inds) / self.tile_area

    def area(self, inds):
        if len(inds) == 0:
            return 0
        else:
            return np.mean([self.objs[ind].area() for ind in inds])
    
    def eccentricity(self, inds):
        if len(inds) == 0:
            return 0
        else:
            return np.mean([self.objs[ind].eccentricity() for ind in inds])

    def circularity(self, inds):
        if len(inds) == 0:
            return 0
        else:
            return np.mean([self.objs[ind].rho() for ind in inds])
        
    def intensity(self, inds):
        if len(inds) == 0:
            return 0
        else:
            return np.mean([np.mean(self.gray.get_tile(*self.objs[ind].bounding_box())) for ind in inds])
