
# system packages

# installed packages
import numpy as np
from numba import jit
from shapely import geometry

# custom packages

# custom Exceptions
ERR = "---> %s <---"
ERR_RESCALE = ERR % ("Cannot rescale data in micron units")
ERR_MICRONS = ERR % ("Data is already in micron units")
ERR_PIXELS = ERR % ("Data is already in pixel units")
ERR_COMPARE = ERR % ("Objects are not in the same resolution")
class UnitException(Exception):
    def __init__(self, message):
        self.message = message

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

# calculates the angle of rotation (degrees) given 2 coordinates forming a line
def find_angle(a, b):
    return np.rad2deg(np.arctan2(b[1]-a[1], b[0]-a[0]))

def linearity(x, y):
    n = len(x)
    x, y = np.array(x), np.array(y)
    ss_xy = np.sum(y * x) - n * np.mean(y) * np.mean(x)
    ss_xx = np.sum(x**2) - n * np.mean(x)**2
    m = ss_xy/ss_xx
    b = np.mean(y) - m * np.mean(x)
    f = m * x + b
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - f)**2)
    rms = np.sqrt(np.mean((y-f)**2))
    r_2 = 1 - ss_res/ss_tot

    return rms

# Array conversion class to handle microns, pixel resolutions, and orders
#  -mpp: microns per pixel constant, usually provided by Loader
#  -ds: level downsample factors, usually provided by Loader
class Converter:
    def __init__(self, mpp, ds):
        self.mpp = mpp
        self.ds = ds
    #
    # end of constructor

    # converts an Array to floating point datatype
    def to_float(self, x):
        if x.dtype == np.float:
            return x
        return x.astype(dtype=np.float, copy=False)
    #
    # end of to_float

    # rounds an Array then converts to integer datatype
    def to_int(self, x):
        if x.dtype == np.int:
            return x
        return np.rint(x).astype(np.int)
    #
    # end of to_int

    # rescales an Array to a new pixel resolution
    def rescale(self, x, lvl):
        if x.lvl == lvl:
            return

        # cannot rescale microns
        if x.is_micron:
            raise UnitException(ERR_RESCALE)

        # scale Array by the following factor: (ds / new_ds)^order
        x *= (self.ds[x.lvl] / self.ds[lvl])**x.order
        x.lvl = lvl
    #
    # end of rescale

    # rescales an Array to max pixel resolution, then converts to microns
    def to_microns(self, x):
        if x.is_micron:
            return

        # rescale to max pixel resolution
        self.rescale(x, 0)

        # scale by the microns to pixel constant
        x *= (self.mpp)**x.order
        x.is_micron = True
    #
    # end of to_microns

    # converts an Array to pixels, then rescales to a given resolution
    def to_pixels(self, x, lvl):

        # scale by the pixels to micron constant
        if not x.is_micron:
            x /= (self.mpp)**x.order
            x.is_micron = False

        # rescale to new resolution
        self.rescale(x, lvl)
    #
    # end of to_pixels


#
# end of Converter

# NOTE: this follows the following guide - https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class Array(np.ndarray):
    def __new__(cls, arr, is_micron=True, lvl=0, order=1):
        obj = np.asarray(arr).view(cls)
        obj.is_micron = is_micron
        obj.lvl = lvl
        obj.order = order
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.is_micron = getattr(obj, 'is_micron', None)
        self.lvl = getattr(obj, 'lvl', None)
        self.order = getattr(obj, 'order', None)

    def translate(self, p):
        if self.lvl != p.lvl or self.is_micron != p.is_micron:
            raise UnitException(ERR_COMPARE)
        self -= p

    def rotate(self, c, angle):
        if self.ndim == 1:
            x0, y0 = self[0], self[1]
        else:
            x0, y0 = self[:, 0], self[:, 1]
        cls = type(self)
        xc, yc = c
        th = np.radians(-angle)

        x1 = xc + np.cos(th) * (x0 - xc) - np.sin(th) * (y0 - yc)
        y1 = yc + np.sin(th) * (x0 - xc) + np.cos(th) * (y0 - yc)

        return cls(x1, y1, self.is_micron, self.lvl, self.order)
#
# end of Array

class Point(Array):
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        arr = np.array((x, y), dtype=np.float)
        obj = Array(arr, is_micron, lvl, order).view(cls)
        return obj
#
# end of Point

class Polygon(Array):
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        arr = np.array((x, y), dtype=np.float).T
        obj = Array(arr, is_micron, lvl, order).view(cls)
        return obj

    def area(self):
        x0, y0 = self[:, 0], self[:, 1]
        x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
        return 0.5 * np.abs(np.dot(x0, y1) - np.dot(x1, y0))

    def centroid(self):
        A = self.area()
        x0, y0 = self[:, 0], self[:, 1]
        x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
        cx = np.sum((x0 + x1)*(x0*y1 - x1*y0)) / (6*A)
        cy = np.sum((y0 + y1)*(x0*y1 - x1*y0)) / (6*A)
        c = Point(cx, cy, self.is_micron, self.lvl, self.order)
        d = np.sqrt((c[0] - x0)**2 + (c[1] - y0)**2)
        r = np.max(d)
        return c, r

    def linear_regression(self):
        x, y, n = self[:, 0], self[:, 1], self.shape[0]
        ss_xy = np.sum(y * x) - n * np.mean(y) * np.mean(x)
        ss_xx = np.sum(x**2) - n * np.mean(x)**2
        m = ss_xy/ss_xx
        b = np.mean(y) - m * np.mean(x)
        return m, b

    def bounding_box(self):
        x, y = self[:, 0], self[:, 1]
        x0, y0 = np.min(x), np.min(y)
        x1, y1 = np.max(x), np.max(y)
        loc = Point(x0, y0, self.is_micron, self.lvl, self.order)
        size = Point(x1-x0, y1-y0, self.is_micron, self.lvl, self.order)
        return loc, size

    def filter(self, p):
        if self.is_micron != p.is_micron or self.lvl != p.lvl:
            raise UnitException(ERR_COMPARE)

        x, y = [], []
        for i in range(self.shape[0]):
            if ray_tracing(self[i][0], self[i][1], np.array(p)):
                x.append(self[i][0])
                y.append(self[i][1])
        return Polygon(x, y, self.is_micron, self.lvl, self.order)

    def connect(self):
        if self[0, 0] != self[-1, 0] or self[0, 1] != self[-1, 1]:
            p = np.array([self[0,0], self[0,1]])[None, :]
            return Array(np.concatenate([self, p], axis=0),
                         self.is_micron, self.lvl, self.order)
        else:
            return self

    def to_shapely(self):
        return geometry.Polygon([[self[i,0], self[i,1]] \
                                 for i in range(self.shape[0])])

#
# end of Polygon

#
# end of file
