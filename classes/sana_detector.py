
import os
import sys
import cv2
import numpy as np
from numba import jit
from PIL import Image, ImageDraw

import sana_geo
from sana_frame import Frame

class Detector:
    def __init__(self, mpp, ds, lvl):
        self.mpp = mpp
        self.ds = ds
        self.lvl = lvl

    # detects objects from a masked frame
    def detect(self, frame):

        # use opencv to find the contours in the image
        contours, hierarchies = cv2.findContours(
            frame.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c, h in zip(contours, hierarchies[0]):
            detection = Detection(self.contour_to_polygon(c), h)
            detections.append(detection)
        self.detections = detections

    # filter the detections into groups based on the area and the hierarchy
    def filter(self, min_body_area, min_hole_area):

        # find the body detections
        # NOTE: these are detections without a parent and above the min size
        for d in self.detections:
            if d.hierarchy[3] == -1 and d.polygon.area() > min_body_area:
                d.body = True

        # find the holes in the body detections
        for d in self.detections:
            if not d.body and self.detections[d.hierarchy[3]].body and \
                d.polygon.area() > min_hole_area:
                d.hole = True

    def contour_to_polygon(self, contour):
        x = np.array([float(v[0][0]) for v in contour])
        y = np.array([float(v[0][1]) for v in contour])
        polygon = sana_geo.Polygon(x, y, self.mpp, self.ds,
                                   is_micron=False, lvl=self.lvl)
        polygon.to_microns()
        return polygon

    # TODO: profile this
    def generate_mask(self, size, x=0, y=1, lvl=None):

        # get all the polygons to use for the mask, rescale and shift if needed
        # NOTE: PIL needs the vertices to be tuples
        polys = []
        for d in self.get_bodies():
            if d.polygon.is_micron:
                d.polygon.to_pixels(self.lvl)
            else:
                d.polygon.rescale(self.lvl)
            d.polygon.round()
            polys.append([tuple(v) for v in d.polygon.vertices()])

        size = np.copy(size)
        size = sana_geo.round(size)
        mask = Image.new('L', (size[0], size[1]), x)
        for poly in polys:
            ImageDraw.Draw(mask).polygon(poly, outline=y, fill=y)
        return Frame(np.array(mask)[:, :, None])

    def ray_trace_vertices(self, p1):
        x, y = [], []
        for d in self.detections:
            p0 = d.polygon
            if p0.is_micron:
                p0.to_pixels(p1.lvl)
            else:
                p0.rescale(p1.lvl)
            for v in p0.vertices():
                if sana_geo.ray_tracing(v[0], v[1], p1.vertices()):
                    x.append(v[0])
                    y.append(v[1])
        return sana_geo.Polygon(np.array(x), np.array(y),
                       self.mpp, self.ds, False, self.lvl)

    def get_bodies(self):
        bodies = []
        for d in self.detections:
            if d.body:
                bodies.append(d)
        return bodies
#
# end of Detector

class TissueDetector(Detector):
    def __init__(self, mpp, ds, lvl):
        super().__init__(mpp, ds, lvl)

    def run(self, frame, min_body_area=0, min_hole_area=0):

        # perform the object detection
        self.detect(frame)

        # filter the tissue detections based on the given areas
        self.filter(min_body_area, min_hole_area)
#
# end of TissueDetector

class Detection:
    def __init__(self, polygon, hierarchy):
        self.polygon = polygon
        self.hierarchy = hierarchy

        self.body = False
        self.hole = False

#
# end of Detection

#
# end of file
