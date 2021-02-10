
import numpy as np
from numba import jit

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

@jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

# defines an (x, y) point in terms of microns
# TODO: could instead make a Unit class instead of point class
# TODO: could just expand this to be an array
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

class Polygon:
    def __init__(self, x, y, mpp, ds, is_micron=True, lvl=0):
        self.x = x
        self.y = y
        self.n = len(x)
        self.mpp = mpp
        self.ds = ds
        self.is_micron = is_micron
        self.lvl = lvl

    def area(self):
        x0, x1, y0, y1 = self.x, np.roll(self.x,1), self.y, np.roll(self.y,1)
        return 0.5 * np.abs(np.dot(x0, y1) - np.dot(x1, y0))

    def centroid(self):
        A = self.area(self.x, self.y)
        x0, x1, y0, y1 = self.x, np.roll(self.x,1), self.y, np.roll(self.y,1)
        cx = np.sum((x0 + x1)*(x0*y1 - x1*y0)) / (6*A)
        cy = np.sum((y0 + y1)*(x0*y1 - x1*y0)) / (6*A)
        c = Point(cx, cy, self.mpp, self.ds, self.is_micron, self.lvl)
        d = np.sqrt((c[0] - self.x)**2 + (c[1] - self.y)**2)
        r = np.max(dist)
        return c, r

    def round(self):
        if self.x.dtype == np.int:
            return
        self.x = np.rint(x, out=x).astype(dtype=np.int, copy=False)
        self.y = np.rint(y, out=y).astype(dtype=np.int, copy=False)

    # rescales a point to a new
    def rescale(self, lvl):
        if self.lvl == lvl:
            return

        # make sure we only rescale if we are in pixels
        if self.is_micron:
            raise UnitException(ERR_RESCALE)

        # scale the point by curr_ds / new_ds
        self.x *= self.ds[self.lvl] / self.ds[lvl]
        self.y *= self.ds[self.lvl] / self.ds[lvl]
        self.lvl = lvl

    def to_microns(self):
        if self.is_micron:
            raise UnitException(ERR_MICRONS)

        # rescale to the original resolution, then convert to microns
        self.rescale(0)
        self.x *= self.mpp
        self.y *= self.mpp
        self.is_micron = True

    def to_pixels(self, lvl):
        if not self.is_micron:
            raise UnitException(ERR_PIXELS)

        # convert to pixels, then downscale to the a certain resolution
        self.x /= self.mpp
        self.y /= self.mpp
        self.is_micron = False
        self.rescale(lvl)

    def vertices(self):
        return np.array([[self.x[i], self.y[i]] for i in range(self.n)])
#
# end of Polygon

#
# end of file
