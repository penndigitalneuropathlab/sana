
import sana.image
import numpy as np

def calculate_ao(pos: sana.image.Frame, mask: sana.image.Frame=None, neg: sana.image.Frame=None):
    """
    this function takes calculate the %Area Occupied of an ROI based on the positive pixel classifications
    :param pos: positive pixel classifications
    :param mask: ROI mask, same size as pos
    :param neg: negative pixel classifications, same size as pos
    """
    if not pos.is_binary():
        raise sana.image.DatatypeException("Positive pixel frame must be binary")
    if not mask is None and not mask.is_binary():
        raise sana.image.DatatypeException("ROI frame must be binary")
    if mask is None:
        mask = sana.image.frame_like(pos, np.ones_like(pos.img))
    if neg is None:
        neg = sana.image.frame_like(pos, np.zeros_like(pos.img))

    num = np.sum(pos.img & mask.img & (1-neg.img))
    den = np.sum(mask.img & (1-neg.img))
    if den == 0:
        return 0, 0
    else:
        return num, den

def bin_cortex(frame: sana.image.Frame, mask: sana.image.Frame=None, neg: sana.image.Frame=None, nbins=1):
    """
    this function converts a normalized cortical frame into bins of %AO
    :param frame: input frame which has been normalized to a common cortical space
    :param mask: ROI mask, same size as input frame
    :param neg: negative pixel classifications, same size as input frame
    :param nbins: number of bins to use
    """
    if not frame.is_deformed:
        raise sana.image.DatatypeException("Frame must be deformed using sana.interpolate.grid_sample")
    if not mask is None and not mask.is_deformed:
        raise sana.image.ImageTypeException("ROI mask must be deformed using sana.interpolate.grid_sample")
    if not mask is None and not mask.is_binary():
        raise sana.image.DatatypeException("ROI mask must be binary")
    if mask is None:
        mask = sana.image.frame_like(frame, np.ones_like(frame.img))
    if not neg is None and not neg.is_deformed:
        raise sana.image.ImageTypeException("Negative pixel mask must be deformed using sana.interpolate.grid_sample")
    if not neg is None and not neg.is_binary():
        raise sana.image.DatatypeException("Negative pixel mask must be binary")
    if neg is None:
        neg = sana.image.frame_like(frame, np.zeros_like(frame.img))
        
    # convert the frame to a curve
    num = np.sum(frame.img & mask.img & (1-neg.img), axis=1)
    den = np.sum(mask.img & (1-neg.img), axis=1)
    x = np.arange(0, num.shape[0])

    # apply the binning
    bin_edges = np.linspace(0, x[-1]+1, nbins+1)
    x_bins = np.digitize(x, bin_edges)
    num_bins = np.array([np.mean(num[x_bins == i]) for i in range(1, nbins+1)])
    den_bins = np.array([np.mean(den[x_bins == i]) for i in range(1, nbins+1)])
    return num_bins, den_bins

class Sampler:
    def __init__(self, seed: int, pos: sana.image.Frame, mask: sana.image.Frame=None, neg: sana.image.Frame=None):
        self.rng = np.random.default_rng(seed)
        self.pos = pos
        if mask is None:
            self.mask = sana.image.frame_like(self.pos, np.ones_like(self.pos.img))
        else:
            self.mask = mask
        if neg is None:
            self.neg = sana.image.frame_like(self.pos, np.zeros_like(self.pos.img))
        else:
            self.neg = neg
        if not all([self.pos.is_binary(), self.mask.is_binary(), self.neg.is_binary()]):
            raise sana.image.DatatypeException("Images must be binary.")
        if not (not any([self.pos.is_deformed, self.mask.is_deformed, self.neg.is_deformed]) or all([self.pos.is_deformed, self.mask.is_deformed, self.neg.is_deformed])):
            raise sana.image.ImageTypeException("Images must all be deformed or all not deformed")

    def _subsample(self, x, y, w, h, align_center=True):
        if align_center:
            ctr = sana.geo.Point(x, y, is_micron=False, level=self.pos.level)
            size = sana.geo.point_like(ctr, w, h)
            loc = ctr - size // 2
        else:
            loc = sana.geo.Point(x, y, is_micron=False, level=self.pos.level)
            size = sana.geo.point_like(loc, w, h)
        pos = sana.image.frame_like(self.pos, self.pos.get_tile(loc, size))
        mask = sana.image.frame_like(self.mask, self.mask.get_tile(loc, size))
        neg = sana.image.frame_like(self.neg, self.neg.get_tile(loc, size))
        num, den = calculate_ao(pos=pos, mask=mask, neg=neg)
        return num, den, loc, size, pos, mask, neg

    def subsample_tiles(self, l, N=1, debug=False):
        x = self.rng.integers(0, self.mask.size()[0], N)
        y = self.rng.integers(0, self.mask.size()[1], N)

        nums = []
        dens = []
        locs = []
        sizes = []
        for i in range(N):
            num, den, loc, size, _, _, _  = self._subsample(x[i], y[i], l, l)
            nums.append(num)
            dens.append(den)
            locs.append(loc)
            sizes.append(size)
        return nums, dens, locs, sizes
    
    def subsample_grid(self, l, pct, avoid_adjacent=False, avoid_partial=True):
        w, h = self.mask.size()

        # calculate the grid origin coordinates
        offset_x = (w % l) // 2
        offset_y = (h % l) // 2
        
        # calculate the grid dimensions                
        ncols = np.floor(w / l).astype(int)
        nrows = np.floor(h / l).astype(int)
        
        # get the total amount of pixels within our grid bounds
        loc = sana.geo.Point(offset_x, offset_y, False, self.mask.level)
        size = sana.geo.point_like(loc, w-offset_x, h-offset_y)
        total_area = np.sum(self.mask.get_tile(loc, size))

        # calculate the amount of pixels we want to sample        
        target_area = pct * total_area
        sampled_area = 0

        # stores the tiles in the grid that have been sampled
        sampled = []

        # stores the tiles that are available to be sampled
        available = list(range(ncols*nrows))
        
        nums, dens, locs, sizes = [], [], [], []

        # continue sampling tiles until we've covered enough area
        while sampled_area < target_area and len(available) != 0:

            # get a random available tile
            idx = available[self.rng.integers(0, len(available))]
            available.remove(idx)

            # get the position in the grid
            j = np.floor(idx // ncols)
            i = idx % ncols

            # get the pixel position
            x, y = i*l+offset_x, j*l+offset_y

            # do the sampling
            num, den, loc, size, pos, mask, neg = self._subsample(x, y, l, l, align_center=False)
            if avoid_partial:
                do_sample = np.sum(mask.img) == mask.img.shape[0] * mask.img.shape[1]
            else:
                do_sample = np.sum(mask.img) != 0
            if do_sample:
                sampled.append(idx)
                nums.append(num)
                dens.append(den)
                locs.append(loc)
                sizes.append(size)

                # keep track of the amount of pixels we've sampled
                sampled_area += np.sum(mask.img & (1-neg.img))

                # remove the adjacent tiles, if desired
                if len(available) != 0 and avoid_adjacent:            
                    up = (j-1)*ncols + i
                    down = (j+1)*ncols + i
                    left = j*ncols + i-1
                    right = j*ncols + i+1           
                    if up in available: available.remove(up)
                    if down in available: available.remove(down)
                    if left in available: available.remove(left)
                    if right in available: available.remove(right)

            # ran out of non-adjacent tiles to sample, repopulate with unsampled tiles
            if len(available) == 0 and avoid_adjacent:
                avoid_adjacent = False
                available = [x for x in range(ncols*nrows) if not x in sampled]

        return nums, dens, locs, sizes            
                
    def subsample_ribbon(self, pct):
        if not self.mask.is_deformed:
            raise sana.image.ImageTypeException(" must be deformed using sana.interpolate.grid_sample")
        w, h = self.mask.size()
        ncols = int(w * pct)
        i = self.rng.integers(0, w-ncols, 1)[0]
        return self._subsample(i, 0, ncols, h, align_center=False)

    def subsample_columns(self, pct):
        w, h = self.mask.size()
        ncols = int(w * pct)
        i_s = self.rng.permutation(np.arange(w))[:ncols]
        pos = sana.image.frame_like(self.pos, self.pos.img[:, i_s])
        mask = sana.image.frame_like(self.mask, self.mask.img[:, i_s])
        num, den = calculate_ao(pos, mask)
        return num, den, pos, mask, i_s
