
# installed modules
import numpy as np

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
                    (self.centers[:,1] > y[0]) & (self.centers[:,1] < y[2]) &\
                    (self.areas >= self.min_area)
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
    def deform(self, img, layers, Nsamp):

        # scale the layer annotations to the heatmap resolution
        layers = [x / self.tiler.ds for x in layers]

        # create the mask for each layer
        size = Point(img.shape[1], img.shape[0], False, DEF_LVL)
        layer_masks = [create_mask([l], size, DEF_LVL, DEF_CONVERTER).img for l in layers]

        # new height is the sum of the num samps per layer
        nh = sum(Nsamp)

        # width is also resampled
        ds = int(nh / img.shape[0])
        nw = ds * img.shape[1]

        # prepare the array for the deformed imag
        deform = np.zeros((nh, nw))

        # loop through the columns
        for j in range(img.shape[1]):

            # loop through the layers
            for i, layer_mask in enumerate(layer_masks):

                # get the indices for a rectangle of data for this column and layer
                x0, x1 = j*ds, j*ds+ds
                y0, y1 = sum(N[:i]), sum(N[:i+1])

                # get the positive indices in the layer mask
                inds = np.where(layer_mask[:,j] != 0)[0]

                # no signal in this case, skip this layer
                if len(inds) == 0:
                    deform[y0:y1, x0:x1] = 0
                    continue

                # extract the signal for this column and layer
                i0, i1 = inds[0], inds[-1]
                sig = img[i0:i1, j]

                # signal was empty, skip this layer
                if len(sig) == 0:
                    deform[y0:y1, x0:x1] = 0

                # only had one sample, just fill the array with the sample
                elif len(sig) == 1:
                    deform[y0:y1, x0:x1] = sig[0]

                # interpolate and place into the deformed heatmap
                else:
                    x = np.arange(sig.shape[0])
                    interp_func = interp1d(x, sig)
                    deform[y0:y1, x0:x1] = np.tile(
                        interp_func(np.linspace(0, sig.shape[0]-1, N[i])),
                        (x1-x0,1)).T
            #
            # end of layer loop
        #
        # end of column loop

        return deform
    #
    # end of deform
