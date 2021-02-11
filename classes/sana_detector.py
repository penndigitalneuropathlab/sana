
import os
import sys
import cv2
import numpy as np
from numba import jit

import sana_geo
from sana_frame import Frame

class Detector:
    def __init__(self, loader):
        self.loader = loader

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
        polygon = sana_geo.Polygon(x, y, self.loader.mpp, self.loader.ds,
                                   is_micron=False, lvl=self.loader.lvl)
        polygon.to_microns()
        return polygon

    # TODO: profile this
    @jit(nopython=True)
    def generate_mask(self, size, x=0, y=1):
        mask = np.full(size, x, dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                for d in self.detections:
                    flag = False
                    if sana_geo.ray_tracing(i, j, d.polygon.vertices()):
                        flag = True
                        break
                if flag:
                    mask[i][j] = y
        return Frame(mask)

    def get_bodies(self):
        bodies = []
        for d in self.detections:
            if d.body:
                bodies.append(d)
        return bodies
#
# end of Detector

class TissueDetector(Detector):
    def __init__(self, loader):
        super().__init__(loader)

    def run(self, frame, min_body_area, min_hole_area):

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
