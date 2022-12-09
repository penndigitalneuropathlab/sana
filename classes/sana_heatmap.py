
# installed modules
import cv2
import numpy as np
from scipy.interpolate import interp1d
# custom modules
from sana_tiler import Tiler
from sana_geo import Point, Polygon

# debugging modules
from matplotlib import pyplot as plt

# TODO: better name, this is kinda a HeatmapGenerator
class Heatmap:
    def __init__(self, frame, objs, tsize, tstep, min_area=0, debug=False):

        self.frame = frame.copy()
        self.gray = frame.copy()
        self.gray.to_gray()
        self.objs = objs
        self.tsize = tsize
        self.tstep = tstep
        self.debug = debug
        self.min_area = min_area
        self.lvl = self.frame.lvl
        self.converter = self.frame.converter

        self.centers = self.get_centers()

        self.areas = self.get_areas()
        self.intensities = self.get_intensities()

        self.tiler = Tiler(self.lvl, self.converter, self.tsize, self.tstep)
        self.tiler.set_frame(self.frame)

        self.tiles = self.tiler.load_tiles()
        self.tile_area = self.tiles[0][0].size # TODO: this is in pixels, is that right?
        
        # calculate the %AO heatmap
        self.ao = np.sum(self.tiles, axis=(2,3)) / self.tile_area
    #
    # end of constructor

    # TODO: is this correct?
    def get_centers(self):
        return np.array([np.mean(obj, axis=0) for obj in self.objs])
    def get_areas(self):
        return np.array([obj.area() for obj in self.objs])
    def get_intensities(self):
        return np.array([self.gray.get_intensity(obj) for obj in self.objs])

    def run(self, funcs):

        feat = np.zeros((len(funcs), self.tiles.shape[0], self.tiles.shape[1]), dtype=float)
        for i, func in enumerate(funcs):
            if not callable(func):
                feat[i, :, :] = func

        if len(self.centers) == 0:
            return feat
                
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
                self.tile_area = tile.area()
                
                # get the objects that are in the tile
                # TODO: add option for other checks, like size for example
                inds = (
                    (self.centers[:,0] > x[0]) & (self.centers[:,0] < x[1]) & \
                    (self.centers[:,1] > y[0]) & (self.centers[:,1] < y[2]) &\
                    (self.areas >= self.min_area)
                )

                # calculate and store the feature
                for n, feat_func in enumerate(funcs):
                    if callable(feat_func):
                        feat[n][i][j] = feat_func(inds)
            #
            # end of i
        #
        # end of j

        return feat
    #
    # end of run

    # TODO: units?
    # TODO: need to decrease size of tile at edges, tile_Area is not consistent
    def density(self, inds):
        return 1000 * np.sum(inds) / self.tile_area

    def area(self, inds):
        if np.sum(inds) == 0:
            return 0
        else:
            a = self.areas[inds]
            mu, sigma = np.mean(a), np.std(a)
            cut_off = sigma * 3
            lo, hi = mu - cut_off, mu + cut_off
            a = a[(a > lo) & (a < hi)]
            if len(a) == 0:
                return 0
            return np.mean(a)

    def intensity(self, inds):
        if np.sum(inds) == 0:
            return 87
        else:
            return np.mean(self.intensities[inds])

    def eccentricity(self, inds):
        if np.sum(inds) == 0:
            return 0
        else:
            return np.mean([self.objs[ind].eccentricity() for ind in inds])

    def circularity(self, inds):
        if np.sum(inds) == 0:
            return 0
        else:
            return np.mean([self.objs[ind].rho() for ind in inds])

    # deforms a heatmap image over the y axis based on the layer annotations
    #  and the number of samples per layer
    # TODO: the transitions are super harsh between layers, might be a bug?
    def deform(self, feats, masks, resize=True):
        Nsamp = 500
        nh = Nsamp // len(masks)

        # scale the masks to the heatmap resolution
        if resize:
            masks = [cv2.resize(x.img, (0,0), fx=1/self.tiler.ds[0], fy=1/self.tiler.ds[1]) \
                     if not x is None else None for x in masks]

        # width is also resampled
        ds = nh / feats.shape[1]
        nw = int(feats.shape[2] * ds)
        
        # prepare the array for the deformed features
        deform = np.zeros((len(masks), feats.shape[0], nh, nw))

        # loop through features
        for feat in range(feats.shape[0]):
            
            # loop through the columns
            for col in range(feats.shape[2]):

                # get the indices for a rectangle of data for this column
                col0, col1 = int(col*ds), int(col*ds+ds)
                row0, row1 = 0, nh
                
                # loop through the layer masks
                for layer, mask in enumerate(masks):
                    if mask is None:
                        continue

                    # get the positive indices in the layer mask
                    inds = np.where(mask[:,col] != 0)[0]

                    # if no signal, skip the layer
                    if len(inds) == 0:
                        continue

                    # extract the signal for this column and layer
                    sig = feats[feat, inds[0]:inds[-1], col]

                    # signal was empty, skip this layer
                    if len(sig) == 0:
                        pass

                    # only had one sample, just fill the array with the sample
                    elif len(sig) == 1:
                        deform[layer, feat, row0:row1, col0:col1] = sig[0]

                    # interpolate and place into the deformed heatmap
                    else:
                        x = np.arange(sig.shape[0])
                        interp_func = interp1d(x, sig)
                        deform[layer, feat, row0:row1, col0:col1] = np.tile(
                            interp_func(np.linspace(0, sig.shape[0]-1, nh)),
                            (col1-col0,1)).T
                #
                # end of mask loop
            #
            # end of column loop
        #
        # end of feat loop

        return deform
    #
    # end of deform
#
# end of Heatmap

#
# end of file
