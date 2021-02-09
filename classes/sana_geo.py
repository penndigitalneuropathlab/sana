
import numpy as np

ERR = "---> %s <---"
ERR_RESCALE = ERR % ("Cannot rescale point in micron units")
ERR_MICRONS = ERR % ("Point is already in micron units")
ERR_PIXELS = ERR % ("Point is already in pixel units")
class UnitException(Exception):
    def __init__(self, message):
        self.message = message

def round(point):
    if point.dtype == np.int:
        return point
    return np.rint(point, out=point).astype(dtype=np.int, copy=False)

# rescales a point to a new
def rescale(point, lvl):
    if point.lvl == lvl:
        return

    # make sure we only rescale points that are in pixel units
    if point.is_micron:
        raise UnitException(ERR_RESCALE)

    # scale the point by curr_ds / new_ds
    point *= (point.ds[point.lvl] / point.ds[lvl])**point.order
    point.lvl = lvl

def to_microns(point):
    if point.is_micron:
        raise UnitException(ERR_MICRONS)

    # rescale to the original resolution, then convert to microns
    rescale(point, 0)
    point *= point.mpp**point.order
    point.is_micron = True

def to_pixels(point, lvl):
    if not point.is_micron:
        raise UnitException(ERR_PIXELS)

    # convert to pixels, then downscale to the a certain resolution
    point /= point.mpp**point.order
    point.is_micron = False
    rescale(point, lvl)

# defines an (x, y) point in terms of microns
# TODO: need a better name than Point
class Point(np.ndarray):
    def __new__(self, x, y, mpp, ds, order=1, is_micron=True, lvl=0):
        self.mpp = mpp
        self.ds = ds
        self.order = order

        self.is_micron = is_micron
        self.lvl = lvl
        return super().__new__(self, shape=(2,), dtype=float,
                               buffer=np.array([x, y], dtype=float))
#
# end of Point

#
# end of file
