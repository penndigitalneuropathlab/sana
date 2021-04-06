
import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from copy import copy
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw
from numba import jit

import sana_io
from sana_geo import Point, Polygon, ray_tracing, find_angle
from sana_color_deconvolution import StainSeparator
import sana_thresholds

from matplotlib import pyplot as plt

# TODO: see where else cv2 can be used
class Frame:
    def __init__(self, img, lvl, converter=None):
        if img.ndim < 3:
            img = img[:, :, None]
        self.img = img
        self.lvl = lvl
        self.converter = converter
        if img.shape[-1] == 3:
            self.color_histo = self.histogram()

    def size(self):
        return np.array((self.img.shape[1], self.img.shape[0]))

    def copy(self):
        return Frame(np.copy(self.img), self.lvl, self.converter)

    def to_gray(self):
        if self.img.shape[-1] == 3:
            float_img = self.img.astype(np.float64)
            self.img = np.dot(float_img, [0.2989, 0.5870, 0.1140])[:, :, None]
            self.gray_histo = self.histogram()

    def to_rgb(self):
        if self.img.shape[-1] == 1:
            self.img = np.tile(self.img, 3)

    def round(self):
        self.img = np.rint(self.img).astype(np.uint8)

    def histogram(self):
        self.round()
        histogram = np.zeros((256, self.img.shape[-1]))
        for i in range(histogram.shape[-1]):
            histogram[:, i] = np.histogram(self.img[:, :, i],
                                           bins=256, range=(0, 255))[0]
        return histogram

    def crop(self, loc, size):
        if loc.is_micron:
            self.converter.to_pixels(loc)
            self.converter.to_pixels(size)
        self.converter.rescale(loc, self.lvl)
        self.converter.rescale(size, self.lvl)
        loc = np.rint(loc).astype(np.int)
        size = np.rint(size).astype(np.int)
        return Frame(self.img[loc[1]:loc[1]+size[1], loc[0]:loc[0]+size[0]],
                     self.lvl, self.converter)

    def rescale(self, ds, size=None):
        img = np.kron(self.img[:, :, 0], np.ones((ds, ds), dtype=np.uint8))
        img = img[:, :, None]

        # NOTE: sometimes the rounding is off by a pixel
        # TODO: make sure this doesn't cause alignment issues
        if size is not None:
            img = img[:size[0], :size[1]]

        return Frame(img, self.lvl, self.converter)

    # NOTE: reshape=False might cause alignment issue down the line
    def rotate(self, angle):
        img = ndimage.rotate(self.img, angle, reshape=False, mode='nearest')
        return Frame(img, self.lvl, self.converter)

    # NOTE: changing the mode affects the GM detection
    #        - 'wrap' causes sharp peaks at edge if the tissue isn't parallel
    #        - 'symmetric' seems to be the best, the edges are very flat
    def pad(self, pad):
        before = pad//2
        after = pad - before
        return Frame(np.pad(self.img,
                            ((before[1],after[1]),(before[0],after[0]),(0,0)),
                            mode='symmetric'), self.lvl, self.converter)

    def save(self, fname):
        sana_io.create_directory(fname)
        if self.img.ndim == 3 and self.img.shape[2] == 1:
            im = self.img[:, :, 0]
        else:
            im = self.img
        im = Image.fromarray(im)
        im.save(fname)

    # calculates the background color as the most common color
    #  in the grayscale space
    def get_bg_color(self):
        self.to_gray()
        histo = self.histogram()
        return np.tile(np.argmax(histo), 3)

    # apply a binary mask to the image
    def mask(self, mask, value=None):
        self.img = cv2.bitwise_and(
            self.img, self.img, mask=mask.img)[:, :, None]
        if value is not None:
            self.img[self.img == 0] = value

    # TODO: use cv2
    # TODO: define sigma in terms of microns
    def gauss_blur(self, sigma):
        self.img = gaussian_filter(self.img, sigma=sigma)
        self.blur_histo = self.histogram()

    def threshold(self, threshold, x, y):

        # prepare img arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        self.img = np.where(self.img < threshold, x, y)

    def get_tissue_threshold(self, blur=0, mi=0, mx=255):
        self.gauss_blur(blur)
        data = self.img.flatten()
        data = data[data >= mi]
        data = data[data <= mx]
        # data = data[::500]
        data = data[:, None]
        return sana_thresholds.kittler(data)[0]
        # means, vars = sana_thresholds.gmm(data, 8, 2)
        # return sana_thresholds.mle(means, vars)[0]

    def get_stain_threshold(self, tissue_mask, blur=0, mi=0, mx=255):
        self.to_gray()
        self.gauss_blur(blur)
        self.mask(tissue_mask, value=0)
        data = self.img.flatten()
        data = data[data != 0]
        data = data[data >= mi]
        data = data[data <= mx]
        data = data[:, None]

        return sana_thresholds.kittler(data)[0]

    def tissue_angle(self, threshold, roi, min_body_area=0, min_hole_area=0):

        # get the filtered tissue detections
        self.detect_tissue(threshold, min_body_area, min_hole_area)

        # find only the detections within the given annotation
        layer_0 = self.ray_trace_detections(roi)
        if layer_0.shape[0] == 0:
            return None

        # perform linear regression and get the line of best fit
        m0, b0 = layer_0.linear_regression()
        a = Point(0, m0*0 + b0, False, self.lvl)
        b = Point(self.size()[0], m0*self.size()[0] + b0, False, self.lvl)
        angle = find_angle(a, b)

        # transform the III and IV quadrants to I and II respectively
        quadrant = angle//90
        if quadrant > 1:
            angle -= 180

        # rotate layer 0 detection, modify angle so that layer 0 is at the
        #  top of the frame instead of the left or right
        layer_0_rot = layer_0.rotate(layer_0.centroid()[0], angle)
        if np.mean(layer_0_rot[:, 1]) >= self.size()[1]//2:
            angle -= 180
        return angle

    def detect(self):
        contours, hierarchies = cv2.findContours(
            self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.detections = []
        for c, h in zip(contours, hierarchies[0]):
            self.detections.append(Detection(c, h, self.lvl, self.converter))

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

    def ray_trace_detections(self, p1):
        x, y = [], []
        for d in self.get_bodies():
            p0 = d.polygon
            if p0.is_micron:
                self.converter.to_pixels(p0, p1.lvl)
            else:
                self.converter.rescale(p0, p1.lvl)
            for i in range(p0.shape[0]):
                if ray_tracing(p0[i][0], p0[i][1], np.array(p1)):
                    x.append(p0[i][0])
                    y.append(p0[i][1])
        x, y = np.array(x), np.array(y)
        return Polygon(x, y, False, self.lvl)

    def get_bodies(self):
        bodies = []
        for d in self.detections:
            if d.body:
                bodies.append(d)
        return bodies

    def detection_mask(self, x=0, y=1):
        polys = []
        for d in self.get_bodies():
            p = copy(d.polygon)
            if p.is_micron:
                self.converter.to_pixels(p, self.lvl)
            else:
                self.converter.rescale(p, self.lvl)
            polys.append([tuple(p[i]) for i in range(p.shape[0])])
        mask = Image.new('L', (self.size()[0], self.size()[1]), x)
        for poly in polys:
            ImageDraw.Draw(mask).polygon(poly, outline=y, fill=y)
        self.img = np.array(mask)[:, :, None]

    def detect_tissue(self, threshold, min_body_area=0, min_hole_area=0):

        # detect the objects on the thresholded image
        self.to_gray()
        self.gauss_blur(1)
        self.threshold(threshold, x=255, y=0)
        self.detect()

        # filter the detections based on the given areas
        self.filter(min_body_area, min_hole_area)

    # NOTE: this should be done on the tissue mask frame
    # TODO: detections may need to be shifted by the amount of blur?
    def detect_layer_0(self, xvals=None):

        # get the rough infra region
        infra = self.img[:self.img.shape[0]//2, :]

        # loop through the columns on the infra region
        x, y = [], []
        if xvals is None:
            xvals = range(infra.shape[1])
        for i in xvals:

            # last index of zeros in the region is the start of tissue
            inds = np.argwhere(infra[:, i] == 0)
            if len(inds) != 0:
                x.append(i)
                y.append(inds[-1][0])

        return np.array(x), np.array(y)

    # NOTE: this should be done on the stain density frame
    def detect_layer_6(self, threshold):

        # normalize by the maximum
        self.img /= np.max(self.img, axis=0)

        # get the rough supra region
        supra = self.img[self.img.shape[0]//2:, :]

        # loop through the columns in the supra region
        x, y = [], []
        for i in range(supra.shape[1]):

            # first index where supra is below threshold is the end of layer 6
            inds = np.argwhere(supra[:, i] <= threshold)
            if len(inds) != 0:
                x.append(i)
                y.append((self.img.shape[0]//2 + inds[0][0]))
        return np.array(x), np.array(y)
#
# end of Frame

class Detection:
    def __init__(self, contour, hierarchy, lvl, converter):
        self.hierarchy = hierarchy
        self.lvl = lvl
        self.converter = converter
        self.polygon = self.contour_to_polygon(contour)

        self.body = False
        self.hole = False

    def contour_to_polygon(self, contour):
        x = np.array([float(v[0][0]) for v in contour])
        y = np.array([float(v[0][1]) for v in contour])
        polygon = Polygon(x, y, False, self.lvl)
        self.converter.to_microns(polygon)
        return polygon
#
# end of Detection

#
# end of file
