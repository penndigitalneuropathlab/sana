
import numpy as np
from numba import jit

ERR = "---> %s <---"
ERR_RESCALE = ERR % ("Cannot rescale data in micron units")
ERR_MICRONS = ERR % ("Data is already in micron units")
ERR_PIXELS = ERR % ("Data is already in pixel units")
ERR_COMPARE = ERR % ("Objects are not in the same resolution")

class UnitException(Exception):
    def __init__(self, message):
        self.message = message

def area(self, poly):
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
    return 0.5 * np.abs(np.dot(x0, y1) - np.dot(x1, y0))

def centroid(self, poly):
    A = self.area()
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
    cx = np.sum((x0 + x1)*(x0*y1 - x1*y0)) / (6*A)
    cy = np.sum((y0 + y1)*(x0*y1 - x1*y0)) / (6*A)
    c = Array(np.array([cx, cy]), poly.is_micron, poly.lvl, poly.order)
    d = np.sqrt((c[0] - x0)**2 + (c[1] - y0)**2)
    r = np.max(d)
    return c, r

def linear_regression(self, poly):
    x, y, n = poly[: ,0], poly[:, 1], poly.n
    ss_xy = np.sum(y * x) - n * np.mean(y) * np.mean(x)
    ss_xx = np.sum(x**2) - n * np.mean(x)**2
    m = ss_xy/ss_xx
    b = np.mean(y) - m * np.mean(x)
    return m, b

def bounding_box(self, poly):
    x, y = poly[: ,0], poly[:, 1]
    x0, y0 = np.min(x), np.min(y)
    x1, y1 = np.max(x), np.max(y)
    loc = Array(np.array([x0, y0]), poly.is_micron, poly.lvl, poly.order)
    size = Array(np.array([x1-x0, y1-y0]), poly.is_micron, poly.lvl, poly.order)
    return loc, size

def translate(self, x, y):
    if x.lvl != y.lvl or x.is_micron != y.is_micron:
        raise UnitException(ERR_COMPARE)
    x -= y

def rotate(self, p, centroid, angle):
    if p.ndim == 1:
        x0, y0 = p[0], p[1]
    else:
        x0, y0 = p[:, 0], p[:, 1]
    obj = type(p)
    xc, yc = centroid
    th = math.radians(-angle)

    x1 = xc + np.cos(th) * (x0 - xc) - np.sin(th) * (y0 - yc)
    y1 = yc + np.sin(th) * (x0 - xc) + np.cos(th) * (y0 - yc)

    arr = np.array([x1, y1])
    if arr.ndim == 2:
        arr = arr.T
    return Array(arr, p.is_micron, p.lvl, p.order)

def filter(self, poly, filt):
    if poly.is_micron != filt.is_micron or poly.lvl != filt.lvl:
        raise UnitException(ERR_COMPARE)

    x, y = [], []
    for i in range(poly.n):
        if ray_tracing(poly[i][0], poly[i][1], np.array(filt)):
            x.append(poly[i][0])
            y.append(poly[i][1])
    return Array(np.array(x, y).T, poly.is_micron, poly.lvl, poly.order)

def connect(self, poly):
    if poly[0, 0] != poly[-1, 0] or poly[0, 1] != poly[-1, 1]:
        poly = np.concatenate([poly, [poly[0,0], poly[1,1]]], axis=0)
        poly.n += 1
    return poly

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

class Converter:
    def __init__(self, mpp, ds):
        self.mpp = mpp
        self.ds = ds

    def to_float(self, x):
        if x.dtype == np.float:
            return x
        return x.astype(dtype=np.float, copy=False)

    def rescale(self, x, lvl):
        if x.lvl == lvl:
            return
        if x.is_micron:
            raise UnitException(ERR_RESCALE)

        x *= (self.ds[x.lvl] / self.ds[lvl])**x.order
        x.lvl = lvl

    def to_microns(self, x):
        if x.is_micron:
            raise UnitException(ERR_MICRONS)

        self.rescale(x, 0)
        x *= (self.mpp)**x.order
        x.is_micron = True

    def to_pixels(self, x, lvl):
        if not x.is_micron:
            raise UnitException(ERR_PIXELS)

        x /= (self.mpp)**x.order
        x.is_micron = False
        self.rescale(x, lvl)

# NOTE: this follows the following guide - https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class Array(np.ndarray):
    def __new__(self, arr, is_micron=False, lvl=0, order=1):
        obj = super().__new__(
            self, shape=arr.shape, dtype=arr.dtype, buffer=arr)
        obj.is_micron = is_micron
        obj.lvl = lvl
        obj.order = order
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.is_micron = getattr(obj, 'is_micron', None)
        self.lvl = getattr(obj, 'lvl', None)
        self.order = getattr(obj, 'order', None)
#
# end of Array

#
# end of file
