
# system packages

# installed packages
import numpy as np
import shapely.geometry

class Converter:
    """
    Unit conversion class which handles conversion between micron and pixel values stored within an Array object (or it's subclasses). Most SANA code will support both microns and pixel units, however usually it will convert to pixels during processing, therefore best practice would be to use pixel units as much as possible
    :param mpp: microns per pixel constant
    :param ds: array of pixel resolution downsample constants
    """
    def __init__(self, mpp=None, ds=None):

        # default values from the PDNL Aperio scanner
        if mpp is None:
            self.mpp = 0.5045
            self.ds = np.array([1.0, 4.0, 16.003152252303728])
        else:
            self.mpp = mpp
            self.ds = ds

    def to_float(self, x):
        """
        Converts an Array to floating point
        :param x: sana.geo.Array
        """
        return x.astype(dtype=float, copy=False)
    
    def to_int(self, x):
        """
        Converts an Array to integer
        :param x: sana.geo.Array
        """
        return x.astype(dtype=int, copy=False)
    
    def rescale(self, x, level):
        """
        Rescales the Array to a new pixel resolution using the following equation:
        $y = x{\frac{ds_current}{ds_new}}^{order}$
        :param x: sana.geo.Array
        :param level: pixel resolution level to rescale to
        """
        if x.is_micron:
            raise UnitException("Cannot rescale data in micron units")
        
        # rescale the Array
        x *= (self.ds[x.level] / self.ds[level])**x.order
        x.level = level

    def to_microns(self, x):
        """
        Converts the Array from pixels to microns. This is done by converting the pixels to the original resolution, then to microns.
        $y = x{\frac{ds_current}{ds_original}}^{order}{mpp}^{order}$
        :param x: sana.geo.Array
        """
        if x.is_micron:
            return
        
        # rescale to original pixel resolution
        self.rescale(x, 0)

        # apply the microns per pixel constant
        x *= (self.mpp)**x.order
        x.is_micron = True

    def to_pixels(self, x, level):
        """
        Converts the Array from microns to pixel units at a given resolution level.
        $y = \frac{x}{{mpp}^{order}}{\frac{ds_original}{ds_new}}^{order}$
        :param x: sana.geo.Array
        :param level: new pixel resolution level
        """

        # apply the microns per pixel constant
        if x.is_micron:
            x /= (self.mpp)**x.order
            x.is_micron = False

        # rescale to new pixel resolution
        self.rescale(x, level)
        

class Array(np.ndarray):
    """
    Wrapper class to a Numpy array, also storing units, resolution, and order. Follows this guide: https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, arr, is_micron=True, level=0, order=1):
        obj = np.asarray(arr).view(cls)
        obj.is_micron = is_micron
        obj.level = level
        obj.order = order
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.is_micron = getattr(obj, 'is_micron', None)
        self.level = getattr(obj, 'level', None)
        self.order = getattr(obj, 'order', None)

    def check_resolution(self, p):
        """
        Ensures the given point is in the same resolution as this Array
        """
        if self.level != p.level or self.is_micron != p.is_micron:
            raise UnitException("Objects are not in the same resolution")

    def translate(self, p):
        """
        Shifts the Array to a new location
        """
        self.check_resolution(p)

        self -= p

    def rotate(self, p, angle):
        """
        Rotates values in the Array counter clockwise around a point 
        :param c: rotation point
        :param angle: angle of rotation in degrees
        """
        self.check_resolution(p)

        x0, y0 = self.get_xy()
        xc, yc = p
        th = np.radians(angle)

        # perform the rotation
        x1 = xc + np.cos(th) * (x0 - xc) - np.sin(th) * (y0 - yc)
        y1 = yc + np.sin(th) * (x0 - xc) + np.cos(th) * (y0 - yc)

        self.set_xy(x1, y1)

def Value(Array):
    """
    (1,) shaped Array object
    :param x: input value
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, x, is_micron=True, level=0, order=1):
        arr = np.array((x,))
        obj = Array(arr, is_micron, level, order).view(cls)
        return obj
    def get_xy(self):
        raise NotImplementedError
    def set_xy(self):
        raise NotImplementedError

class Point(Array):
    """
    (2,) shaped Array object
    :param x: x value
    :param y: y value
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, x, y, is_micron=True, level=0, order=1):
        arr = np.array((x, y))
        obj = Array(arr, is_micron, level, order).view(cls)
        return obj
    
    def get_xy(self):
        return self[0], self[1]
    def set_xy(self, x, y):
        self[0] = x
        self[1] = y

    def transform(self, M, inverse=False):
        """
        Applies a transformation matrix to the Point
        :param M: (2,3) transformation matrix M
        :param inverse: whether or not to inverse the transformation
        """
        if not inverse:
            np.matmul(M[:,:2], self, out=self)
            self += M[:,2]
        else:
            self -= M[:,2]
            np.matmul(np.linalg.inv(M[:,:2]), self, out=self)
    
class Polygon(Array):
    """
    (n,2) shaped Array object, contains specialized functions for Polygons
    :param x: (n,1) array
    :param y: (n,1) array
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, x, y, is_micron=True, level=0, order=1):
        arr = np.array((x, y), dtype=float).T
        obj = Array(arr, is_micron, level, order).view(cls)
        return obj
    
    def get_xy(self):
        return self[:,0], self[:,1]
    def set_xy(self, x, y):
        self[:,0] = x
        self[:,1] = y

    def get_rolled_xy(self):
        """
        Convenience function for rolling the first point to the last
        """
        x, y = self.get_xy()
        return np.roll(x, 1), np.roll(y, 1)

    def transform(self, M, inverse=False):
        """
        Applies a transformation matrix to the Polygon
        :param M: (2,3) transformation matrix M
        :param inverse: whether or not to inverse the transformation
        """
        if not inverse:
            np.matmul(self, M[:,:2].T, out=self)
            self += M[:,2]
        else:
            self -= M[:,2]
            np.matmul(self, np.linalg.inv(M[:,:2].T), out=self)

    def get_area(self):
        """
        Calculates the area of the Polygon
        """
        if not hasattr(self, 'A'):
            x0, y0 = self.get_xy()
            x1, y1 = self.get_rolled_xy()
            A = 0.5 * np.abs(np.dot(x0, y1) - np.dot(x1, y0))
            self.A = value_like(A, self)
            self.A.order = 2
        return self.A
    
    def get_centroid(self):
        """
        Calculates the center of gravity of the Polygon, then gets the radius of the centroid by selecting the max distance vertex from the center
        """
        A = self.get_area()
        x0, y0 = self.get_xy()
        x1, y1 = self.get_rolled_xy()
        cx = np.sum((x0 + x1)*(x0*y1 - x1*y0)) / (6*A)
        cy = np.sum((y0 + y1)*(x0*y1 - x1*y0)) / (6*A)
        c = point_like(cx, cy, self)
        d = np.sqrt((c[0] - x0)**2 + (c[1] - y0)**2)
        r = np.max(d)

        return c, r

    def get_axes(self):
        """
        Calculates the major and minor axes based on the minimum bounding rectangle of the Polygon
        """
        rect = self.get_minimum_bounding_rectangle()
        side_lengths = np.sqrt(np.sum((rect[:-1] - rect[1:])**2, axis=1))
        major = float(np.max(side_lengths)/2)
        minor = float(np.min(side_lengths)/2)
        return major, minor    

    # TODO: anyway to clean this up?
    def get_minimum_bounding_rectangle(self):
        """
        Calculates the minimum bounding rectangle of the Polygon
        """
        pi2 = np.pi/2

        # TODO: does ConvexHull work on Polygon? if not use to_convexhull
        hull = self[ConvexHull(self).vertices]

        edges = hull[1:] - hull[:-1]

        angles = np.arctan2(edges[:,1], edges[:,0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-pi2),
            np.cos(angles+pi2),
            np.cos(angles),
        ]).T
        rotations = rotations.reshape((-1, 2, 2))

        rot_points = np.dot(rotations, hull.T)

        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        areas = (max_x - min_x) * (max_y - min_y)
        best_ind = np.argmin(areas)

        x1 = max_x[best_ind]
        x2 = min_x[best_ind]
        y1 = max_y[best_ind]
        y2 = min_y[best_ind]
        r = rotations[best_ind]

        rect = np.zeros((4,2))
        rect[0] = np.dot([x1, y2], r)
        rect[1] = np.dot([x2, y2], r)
        rect[2] = np.dot([x2, y1], r)
        rect[3] = np.dot([x1, y1], r)

        return polygon_like(rect[:,0], rect[:,1], self)

    def get_major(self):
        """
        Calculates the length of the major axis
        """
        if not hasattr(self, 'major'):
            self.major, self.minor = self.get_axes()
        return self.major
    
    def get_minor(self):
        """
        Calculates the length of the minor axis
        """
        if not hasattr(self, 'minor'):
            self.major, self.minor = self.get_axes()
        return self.minor
    
    def get_eccentricity(self):
        """
        Calculates the eccentricity feature, measuring the "elliptic" nature of the polygon: 0.0 is a circle, 1.0 is a line
        """
        if not hasattr(self, 'eccentricity'):
            major = self.get_major()
            minor = self.get_minor()
            self.eccentricity = np.sqrt(1-minor**2/major**2)
        return self.eccentricity

    def get_perimeter(self):
        """
        Calculates the perimeter feature
        """
        if not hasattr(self, 'perimeter'):
            x0, y0 = self.get_xy()
            x1, y1 = self.get_rolled_xy()
            self.perimeter = np.sum(np.sqrt((y1-y0)**2+(x1-x0)**2))
        return self.perimeter
    
    def get_circularity(self):
        """
        Calculates the circularity feature: 0.0 is a square, 1.0 is a circle
        """
        if not hasattr(self, 'circularity'):
            A = self.get_area()
            perimeter = self.get_perimeter()
            return (4*A*np.pi) / perimeter**2
        
    def bounding_box(self):
        """
        Gets the location and size of the orthogonal bounding box
        """
        x, y = self.get_xy()
        x0, y0 = np.min(x), np.min(y)
        x1, y1 = np.max(x), np.max(y)
        loc = point_like(x0, y0, self)
        size = point_like(x1-x0, y1-y0, self)
        return loc, size
    
    def bounding_centroid(self):
        """
        Gets the bounding box of the centroid
        """
        c, r = self.centroid()
        loc = point_like(c[0]-r, c[1]-r, self)
        size = point_like(2*r, 2*r, self)
        return loc, size
    
    def is_inside(self, poly):
        """
        Checks if all vertices of this Polygon are in the input Polygon
        :param poly: input Polygon
        """
        self.check_resolution(poly)

        # prepare the input Polygon, ray_tracing cannot handle Polygons
        poly = np.array(poly)
        
        for i in range(self.shape[0]):
            if not ray_tracing(self[i][0], self[i][1], np.array(poly)):
                return False
        return True
    
    def is_partially_inside(self, poly):
        """
        Checks if any vertex of this Polygon is inside the input Polygon
        :param poly: input Polygon
        """
        self.check_resolution(poly)

        # prepare the input Polygon, ray_tracing cannot handle Polygons
        poly = np.array(poly)

        for i in range(self.shape[0]):
            if ray_tracing(self[i][0], self[i][1], poly):
                return True
        return False
    
    def filter_vertices(self, poly):
        """
        Deletes vertices from this Polygon that are not inside the input Polygon
        :param poly: input Polygon
        """
        self.check_resolution(poly)

        # prepare the input Polygon, ray_tracing cannot handle Polygons
        poly = np.array(poly)

        x, y = [], []
        for i in range(self.shape[0]):
            if ray_tracing(self[i][0], self[i][1], poly):
                x.append(self[i][0])
                y.append(self[i][1])
        return polygon_like(x, y, self)

    def is_connected(self):
        """
        Checks if the first and last vertex are equivalent
        """
        return np.isclose(self[0,0], self[-1,0]) and np.isclose(self[0,1], self[-1,1])
    
    def connect(self):
        """
        Connects the Polygon by copying the first vertex to the last
        """
        if self.is_connected():
            return self
        else:
            x, y = self.get_xy()
            x = np.concatenate([x, [self[0,0]]], axis=0)
            y = np.concatenate([y, [self[0,1]]], axis=0)
            return polygon_like(x, y, self)

    def disconnect(self):
        """
        Disconnects the Polygon by deleting the duplicated final vertex
        """
        if not self.is_connected():
            return self
        else:
            x, y = self.get_xy()
            x = x[:-1]
            y = y[:-1]
            return polygon_like(x, y, self)
        
    def to_shapely(self):
        """
        Converts the Polygon to a Shapely object for access to certain functions not worth implementing
        """
        return shapely.geometry.Polygon([[self[i,0], self[i,1]] \
                                         for i in range(self.shape[0])])
        
    def to_annotation(self, class_name="", annotation_name="", confidence=1.0, connect=True):
        """
        Converts the Polygon to an Annotation
        :param class_name: class name to store
        :param annotation_name: annotation name to store
        :param confidence: confidence of annotation
        :param connect: whether or not to connect the Annotation
        """
        if connect:
            x, y = self.connect().get_xy()
        else:
            x, y = self.get_xy()
        return Annotation(x=x, y=y, geo=None, class_name=class_name, annotation_name=annotation_name, confidence=confidence, is_micron=self.is_micron, level=self.level, order=self.order)

class Line(Polygon):
    """
    Specific instance of a Polygon that is not a closed shape
    :param x: (n,1) array
    :param y: (n,1) array
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, x, y, is_micron=True, lvl=0, order=1):
        obj = Polygon(x, y, is_micron, lvl, order).view(cls)
        return obj@jit(nopython=True)
    
    def linear_regression(self):
        """
        Calculates y = mx + b line of best fit
        """
        x, y = self.get_xy()
        n = self.shape[0]
        ss_xy = float(np.sum(y * x) - n * np.mean(y) * np.mean(x))
        ss_xx = float(np.sum(x**2) - n * np.mean(x)**2)
        m = ss_xy/ss_xx
        b = np.mean(y) - m * np.mean(x)
        return m, b
    
    def get_angle(self):
        """
        Calculates the angle of rotation in degrees based on the line of best fit
        """

        # calculate the line of best fit
        m, b = self.linear_regression()
        a = Point(self[0,0], m*self[0,0] + b, False, self.lvl)
        b = Point(self[-1,0], m*self[-1,0] + b, False, self.lvl)

        # calculate the angle of rotation in degrees
        angle = np.rad2deg(np.arctan(m))

        # get only positive angles
        if angle < 0:
            angle = 360 + angle

        # only care about angles in quadrants I and II
        if angle > 180:
            angle -= 180

        return angle
    
    def to_annotation(self, file_name, class_name,
                      anno_name="", confidence=1.0, connect=True):
        """
        Converts the Line to an Annotation using the LineString format in geojson
        """
        return Annotation(x=x, y=y, geo=None, class_name=class_name, annotation_name=annotation_name, confidence=confidence, is_micron=self.is_micron, level=self.level, order=self.order, object_type='LineString')
    
class Annotation(Polygon):
    """
    Special case of a Polygon that stores extra information about the object, usually from a geojson formatted file.
    :param geo: geojson formatted vertices, not needed if providing x and y parameters
    :param x: (n,1) input array, not needed if providing geo
    :param y: (n,1) input array, not needed if providing geo
    :param class_name: class name of the object in the geojson
    :param annotation_name: name of the object in the geojson
    :param confidence: floating point value, usually from a machine learning model
    :param object_type: string denoting the type of object to use in a geojson file format
    :param is_micron: flag denoting if the Array is micron units or pixel units
    :param level: pixel resolution level (level=0 when in microns)
    :param order: order of the data, usually used for calculating areas
    """
    def __new__(cls, geo, x=None, y=None, class_name="", annotation_name="", confidence=1.0, object_type='Polygon', is_micron=True, lvl=0, order=1):
    
        # initalize the array using the geometry
        if x is None or y is None:
            x, y = cls.get_vertices(geo)
        obj = Polygon(x, y, is_micron, lvl, order).view(cls)
        
        # store attributes
        obj.class_name = class_name
        obj.annotation_name = annotation_name
        obj.confidence = confidence
        obj.object_type = object_type
    
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.class_name = getattr(obj, 'class_name', None)
        self.annotation_name = getattr(obj, 'annotation_name', None)
        self.confidence = getattr(obj, 'confidence', None)
        self.is_micron = getattr(obj, 'is_micron', None)
        self.lvl = getattr(obj, 'lvl', None)
        self.order = getattr(obj, 'order', None)
        self.object_type = getattr(obj, 'object_type', None)

    # TODO: simplify or use geojson
    # TODO: make a MultiPolygon object  
    # TODO: move this out of geo.py and into io.py
    def get_vertices(geo):
        """
        Generates a set of x,y coordinates from geojson formatted vertices
        """
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
        elif geo['type'] == 'LineString':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords])
            y = np.array([float(c[1]) for c in coords])            
        else:
            x = np.array([])
            y = np.array([])
        #
        # end of geo type checking

        return x, y
    
    # TODO: rename to geojson?
    def to_json(self):
        """
        Generates a formatted geojson dictionary
        """

        # generate a list of vertices from the Array
        verts = []
        for i in range(self.shape[0]):
            verts.append([self[i][0], self[i][1]])
        if self.object_type == 'Polygon':
            verts = [verts]
            
        # create the JSON format, using the given class and name
        annotation = {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
                "type": self.object_type,
                "coordinates": verts,
            },
            "properties": {
                "name": self.name,
                "objectType": "annotation",
                "classification": {
                    "name": self.class_name,
                    "color": [255, 0, 0],
                },
                "confidence": self.confidence,
            }
        }
        return annotation

    def connect(self):
        if not self.connected():
            poly = super().connect()
            return poly.to_annotation(self.file_name, self.class_name, self.name, self.confidence, connect=True)
        else:
            return self
        
    def disconnect(self):
        if self.connected():
            poly = super().disconnect()
            return poly.to_annotation(self.file_name, self.class_name, self.name, self.confidence, connect=False)
        else:
            return self

def ray_tracing(x,y,poly):
    """
    Detects if the given xy point is inside the poly array. This uses numba so that the loops are very fast, therefore the inputs must be basic C datatypes
    :param x: int
    :param y: int
    :param poly: (n,2) numpy array
    """
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

def array_like(arr, obj):
    return Array(arr, is_micron=obj.is_micron, level=obj.level, order=obj.order)
def value_like(x, obj):
    return Value(x, is_micron=obj.is_micron, level=obj.level, order=obj.order)
def point_like(x, y, obj):
    return Point(x, y, is_micron=obj.is_micron, level=obj.level, order=obj.order)
def polygon_like(x, y, obj):
    return Polygon(x, y, is_micron=obj.is_micron, level=obj.level, order=obj.order)
def line_like(x, y, obj):
    return Line(x, y, is_micron=obj.is_micron, level=obj.level, order=obj.order)

class UnitException(Exception):
    def __init__(self, message):
        self.message = message