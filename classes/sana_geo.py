
# system packages

# installed packages
import numpy as np
from numba import jit
from shapely import geometry
from matplotlib import pyplot as plt

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

# detects if the given xy point is inside the poly array
# NOTE: this needs to be basic C datatypes so that numba can optimize it
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
#
# end of ray_tracing

# this function assumes we are processing a segmentation which is essentially
#  the joining of 2 boundary annotations. It finds the max distance between adjacent
#  vertices to separate into the 2 annotations
def separate_seg(x, y_only=False):
    dist = []
    for i in range(x.shape[0]-1):
        a, b = x[i], x[i+1]
        if not y_only:
            score = (a[0] - b[0])**2 + (a[1] - b[1])**2
        else:
            score = (a[1] - b[1])**2
        dist.append(score)
    dist = np.array(dist)

    # get the 2 separation vertices
    seps = sorted(list((-dist).argsort()[:2]))

    # split the segmentation into 2 annotations
    x0, x1 = [], []
    for i in range(0, seps[0]+1):
        x0.append(x[i])
    for i in range(seps[0]+1, seps[1]+1):
        x1.append(x[i])
    for i in range(seps[1]+1, x.shape[0]):
        x0.append(x[i])
    x0 = np.array(sorted(x0, key=lambda x: x[0]))
    x1 = np.array(sorted(x1, key=lambda x: x[0]))

    x0 = Polygon(x0[:, 0], x0[:, 1], x.is_micron, x.lvl, x.order)
    x1 = Polygon(x1[:, 0], x1[:, 1], x.is_micron, x.lvl, x.order)
    return [x0, x1]
#
# end of separate_seg

# this function separates the seg into 2 parts, then finds the angle
# that matches the x extrema of each boundary seg as closely as possible
def get_ortho_angle(seg):
    s0, s1 = separate_seg(seg)

    # loop through angles to test
    center = seg.centroid()[0]
    angles = np.arange(-180, 180, 0.1)
    dist = []
    for i in angles:

        # rotate by the current angle
        x, y = s0.copy(), s1.copy()
        x.rotate(center, i)
        y.rotate(center, i)

        # get the distance between the x extrema
        x0, x1 = np.min(x[:,0]), np.max(x[:,0])
        y0, y1 = np.min(y[:,0]), np.max(y[:,0])
        dist.append(np.sqrt(((x0-y0)**2)+((x1-y1)**2)))

    # get the best angle to use
    return angles[np.argmin(dist)]
#
# end of get_ortho_seg

# performs a series a translations and rotations to transform a polygon
#  to the coordinate system of a processed Frame
def transform_poly(x, loc, crop_loc, M1, M2):
    if not loc is None:
        x.translate(loc)
    if not M1 is None:
        x = x.transform(M1)
    if not crop_loc is None:
        x.translate(crop_loc)
    if not M2 is None:
        x = x.transform(M2)
    return x
#
# end of transform_poly

# performs the inverse translations and rotations to return a polygon
#  to the original coordinate system
def transform_inv_poly(x, loc, crop_loc, M1, M2):
    if not M2 is None:
        x = x.transform_inv(M2)
    if not crop_loc is None:
        x.translate(-crop_loc)
    if not M1 is None:
        x = x.transform_inv(M1)
    if not loc is None:
        x = x.translate(-loc)
    return x
#
# end of transform_inv_poly

# VERY useful function for plotting a polygon onto a axis
def plot_poly(ax, x, color='black', last=True, linestyle='-', label=None):
    if last:
        en = x.shape[0]-1
    else:
        en = x.shape[0]-2
    for i in range(en):
        if i != 0:
            label = None
        ax.plot((x[i][0], x[i+1][0]),
                (x[i][1], x[i+1][1]),
                color=color, linestyle=linestyle, label=label)            

# converts a Convexhull into a polygon
def hull_to_poly(hull, xy, lvl=0):
    x, y = [], []
    for v in hull.vertices:
        v = xy[v]
        x.append(v[0])
        y.append(v[1])
    x, y = np.array(x), np.array(y)
    p = Polygon(x, y, False, lvl)
    poly = p.connect()
    return poly
#
# end of hull_to_pull

# Array conversion class to handle microns, pixel resolutions, and orders
# NOTE: SANA code supports both microns and pixel analysis, but it is best
#       practice to use pixel units as much as possible, and convert to/from
#       microns at the beginning or ending of analysis
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
        if x.is_micron:
            x /= (self.mpp)**x.order
            x.is_micron = False

        # rescale to new resolution
        self.rescale(x, lvl)
    #
    # end of to_pixels
#
# end of Converter

# wrapper class to a Numpy array, also stores the units, resolution, and order
# NOTE: this follows the following guide - https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
#  -is_micron: defines if the units are either microns or pixels
#  -lvl: pixel resolution of the data (lvl=0 if in microns)
#  -order: order of the data, usual for calculating areas
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
    #
    # end of constructor

    # shifts the array to a new location
    # NOTE: this is often used to move to the origin
    #        e.g. loc, size = x.bounding_box(); x.translate(loc)
    def translate(self, p):
        if self.lvl != p.lvl or self.is_micron != p.is_micron:
            raise UnitException(ERR_COMPARE)
        self -= p
    #
    # end of translate

    # rotates all points in the Array around a center point by angle in degrees
    #  -c: center point to rotate around
    #  -angle: angle of rotation in degrees
    def rotate(self, c, angle):
        if self.lvl != c.lvl or self.is_micron != c.is_micron:
            raise UnitException(ERR_COMPARE)

        # get the xy values
        if self.ndim == 1:
            x0, y0 = self[0], self[1]
        else:
            x0, y0 = self[:, 0], self[:, 1]

        # prepare the center and angle in radians
        xc, yc = c
        th = np.radians(-angle)

        # perform the rotation
        x1 = xc + np.cos(th) * (x0 - xc) - np.sin(th) * (y0 - yc)
        y1 = yc + np.sin(th) * (x0 - xc) + np.cos(th) * (y0 - yc)

        # set the xy values
        if self.ndim == 1:
            self[0] = x1
            self[1] = y1
        else:
            self[:, 0] = x1
            self[:, 1] = y1
    #
    # end of rotate
#
# end of Array

# simpler inheritance of Array to simplify usage
# NOTE: shape of this Array is guaranteed to be (2,)
class Point(Array):
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        arr = np.array((x, y), dtype=np.float)
        obj = Array(arr, is_micron, lvl, order).view(cls)
        return obj
    #
    # end of Constructor
#
# end of Point

# inherits Array for more specialized functions and uses
# NOTE: shape of this Array is guaranteed to be (N,2)
#  -x: (N, 1) array
#  -y: (N, 1) array
class Polygon(Array):
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        arr = np.array((x, y), dtype=np.float).T
        obj = Array(arr, is_micron, lvl, order).view(cls)
        return obj

    # separates the Polygon into x and y arrays
    def get_xy(self):
        if not hasattr(self, 'x'):
            self.x = self[:,0]
            self.y = self[:,1]
        return self.x, self.y
    #
    # end of get_xy

    def get_rolled_xy(self):
        x0, y0 = self.get_xy()
        if not hasattr(self, 'x'):
            self.x1 = np.roll(x0, 1)
            self.y1 = np.roll(y0, 1)
        return self.x1, self.y1

    # TODO: make this inplace
    def transform(self, M):
        x = self @ M[:,:2].T + M[:,2]
        return Polygon(x[:,0], x[:,1], self.is_micron, self.lvl, self.order)
    def transform_inv(self, M):
        x = (self - M[:,2]) @ np.linalg.inv(M[:,:2].T)
        return Polygon(x[:,0], x[:,1], self.is_micron, self.lvl, self.order)

    # calculates the area of the Polygon
    # TODO: this should return a Value (or something) that tracks the order
    def area(self):
        x0, y0 = self.get_xy()
        x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
        return 0.5 * np.abs(np.dot(x0, y1) - np.dot(x1, y0))
    #
    # end of area

    # gets the center of gravity of the Polygon, then gets the radius of the
    # centroid by selecting the max distance vertex from the center
    # TODO: this needs to follow what area does
    def centroid(self):
        A = self.area()
        x0, y0 = self.get_xy()
        x1, y1 = np.roll(x0, 1), np.roll(y0, 1)
        cx = np.sum((x0 + x1)*(x0*y1 - x1*y0)) / (6*A)
        cy = np.sum((y0 + y1)*(x0*y1 - x1*y0)) / (6*A)
        c = Point(cx, cy, self.is_micron, self.lvl, self.order)
        d = np.sqrt((c[0] - x0)**2 + (c[1] - y0)**2)
        r = np.max(d)
        return c, r
    #
    # end of centroid

    # gets a bounding rectangle based on the centroid of the Polygon
    def bounding_centroid(self):
        c, r = self.centroid()
        loc = Point(c[0]-r, c[1]-r, self.is_micron, self.lvl, self.order)
        size = Point(2*r, 2*r, self.is_micron, self.lvl, self.order)
        return loc, size
    #
    # end of bounding_centroid

    # gets a bounding rectangle based on the extrema of the Polygon
    def bounding_box(self):
        x, y = self[:, 0], self[:, 1]
        x0, y0 = np.min(x), np.min(y)
        x1, y1 = np.max(x), np.max(y)
        loc = Point(x0, y0, self.is_micron, self.lvl, self.order)
        size = Point(x1-x0, y1-y0, self.is_micron, self.lvl, self.order)
        return loc, size
    #
    # end of bounding_box

    def filter(self, p):
        if self.is_micron != p.is_micron or self.lvl != p.lvl:
            raise UnitException(ERR_COMPARE)

        x, y = [], []
        for i in range(self.shape[0]):
            if ray_tracing(self[i][0], self[i][1], np.array(p)):
                x.append(self[i][0])
                y.append(self[i][1])
        return Polygon(x, y, self.is_micron, self.lvl, self.order)

    def inside(self, p):
        if self.is_micron != p.is_micron or self.lvl != p.lvl:
            raise UnitException(ERR_COMPARE)

        for i in range(self.shape[0]):
            if not ray_tracing(self[i][0], self[i][1], np.array(p)):
                return False
        return True

    def connected(self):
        return self[0, 0] == self[-1, 0] and self[0, 1] == self[-1, 1]
    
    # TODO: this and filter need to return Annotation sometimes...
    def connect(self):
        if not self.connected():
            x, y = self.get_xy()
            x = np.concatenate([x, [self[0,0]]], axis=0)
            y = np.concatenate([y, [self[0,1]]], axis=0)
            return Polygon(x, y, self.is_micron, self.lvl, self.order)
        else:
            return self

    # converts the Array to a Shapely object to access certain functions
    def to_shapely(self):
        return geometry.Polygon([[self[i,0], self[i,1]] \
                                 for i in range(self.shape[0])])
    #
    # end of to_shapely

    # convert the Array to a Annotation to prepare for file io
    def to_annotation(self, file_name, class_name,
                      anno_name="", confidence=1.0):
        x, y = self.connect().get_xy()
        return Annotation(None, file_name, class_name, anno_name,
                          confidence=confidence, is_micron=self.is_micron,
                          lvl=self.lvl, order=self.order, x=x, y=y)
    #
    # end of to_annotation

    def clip(self, x0, x1, y0, y1):
        x, y = [], []
        for i in range(self.shape[0]):
            if self[i][0] <= x1 and self[i][0] >= x0 and \
                self[i][1] <= y1 and self[i][1] >= y0:
                x.append(self[i][0])
                y.append(self[i][1])
        return Polygon(x, y, self.is_micron, self.lvl, self.order)
#
# end of Polygon

class Line(Polygon):
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        obj = Polygon(x, y, is_micron, lvl, order).view(cls)
        return obj
    #
    # end of constructor

    # calculates the angle of rotation in degrees based on the linear regression
    def get_angle(self):

        # generate a linear represntation
        m, b = self.linear_regression()
        a = Point(self[0,0], m*self[0,0] + b, False, self.lvl)
        b = Point(self[-1,0], m*self[-1,0] + b, False, self.lvl)

        # calculate the angle of rotation in degrees
        angle = np.rad2deg(np.arctan(m))

        # get only pos. angles
        if angle < 0:
            angle = 360 + angle

        # only care about angles in quadrants I and II
        if angle > 180:
            angle -= 180

        return angle
    #
    # end of get_angle

    # calculates the line of best fit
    def linear_regression(self):
        x, y, n = self[:, 0], self[:, 1], self.shape[0]
        ss_xy = float(np.sum(y * x) - n * np.mean(y) * np.mean(x))
        ss_xx = float(np.sum(x**2) - n * np.mean(x)**2)
        m = ss_xy/ss_xx
        b = np.mean(y) - m * np.mean(x)
        return m, b

class Annotation(Polygon):
    def __new__(cls, geo, file_name, class_name, anno_name, confidence=1.0,
                is_micron=True, lvl=0, order=1, x=None, y=None):

        # initalize the array using the geometry
        if x is None or y is None:
            x, y = cls.get_vertices(geo)
        obj = Polygon(x, y, is_micron, lvl, order).view(cls)

        # store attributes
        obj.file_name = file_name
        obj.class_name = class_name
        obj.name = anno_name
        obj.confidence = confidence

        return obj
    #
    # end of constructor

    # generates a set of xy coordinates from the geometry
    def get_vertices(geo):

        # TODO: this should be simplified.
        #        need to actually handle what a MultiPolygon is
        if geo['type'] == 'MultiPolygon':
            coords_list = geo['coordinates']
            x, y = [], []
            for coords in coords_list:
                x += [float(c[0]) for c in coords[0]]
                y += [float(c[1]) for c in coords[0]]
            x = np.array(x)
            y = np.array(y)
        elif geo['type'] == 'Polygon':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords[0]])
            y = np.array([float(c[1]) for c in coords[0]])
        elif geo['type'] == 'MultiPoint':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords])
            y = np.array([float(c[1]) for c in coords])            
        else:
            x = np.array([])
            y = np.array([])
        #
        # end of geo type checking

        return x, y
    #
    # end of get_xy

    # returns a json string representation of the object
    def to_json(self):

        # generate a list of vertices from the Array
        verts = []
        for i in range(self.shape[0]):
            verts.append([self[i][0], self[i][1]])

        # create the JSON format, using the given class and name
        annotation = {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
                "type": "Polygon",
                "coordinates": [verts]
            },
            "properties": {
                "name": self.name,
                "classification": {
                    "name": self.class_name,
                },
                "confidence": self.confidence,
            }
        }
        return annotation
    #
    # end of anno_to_json

    def connect(self):
        if not self.connected():
            x, y = self.get_xy()
            x = np.concatenate([x, [self[0,0]]], axis=0)
            y = np.concatenate([y, [self[0,1]]], axis=0)
            p = Polygon(x, y, self.is_micron, self.lvl, self.order)
            return p.to_annotation(self.file_name, self.class_name, self.name, self.confidence)            
        else:
            return self

#
# end of Annotation

#
# end of file
