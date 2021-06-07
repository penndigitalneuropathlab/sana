
import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from copy import copy
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw

import sana_io
from sana_geo import Point, Polygon, ray_tracing, find_angle
from sana_color_deconvolution import StainSeparator
import sana_thresholds

from matplotlib import pyplot as plt

# TODO: see where else cv2 can be used
class Frame:
    def __init__(self, img, lvl=-1, converter=None):
        if img.ndim < 3:
            img = img[:, :, None]
        self.img = img
        self.lvl = lvl
        self.converter = converter
        if img.shape[-1] == 3:
            # self.color_histo = self.histogram()
            pass
    def size(self):
        return np.array((self.img.shape[1], self.img.shape[0]))

    def copy(self):
        return Frame(np.copy(self.img), self.lvl, self.converter)

    def to_gray(self):
        if self.img.shape[-1] == 3:
            float_img = self.img.astype(np.float64)
            self.img = np.dot(float_img, [0.2989, 0.5870, 0.1140])[:, :, None]
            # self.gray_histo = self.histogram()

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

    def rotate(self, angle):
        center = tuple(np.array([self.img.shape[0]//2, self.img.shape[1]//2]))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            self.img, rot_mat, self.img.shape[0:2],
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
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
        if self.img.dtype == np.uint8:
            if self.img.ndim == 3 and self.img.shape[2] == 1:
                im = self.img[:, :, 0]
            else:
                im = self.img
            if np.max(im) == 1:
                im = 255 * im
            im = Image.fromarray(im)
            im.save(fname)
        else:
            np.save(fname.split('.')[0]+'.npy', self.img)

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
        if sigma != 0:
            self.img = gaussian_filter(self.img, sigma=sigma)

    # niter - number of iterations [1, 12]
    # kappa - conduction coefficient (20-100) [50, 11]
    # gamma - max value of .25 for stability [.1, 0.9]
    # step  - distance between adjacent pixels [(1.,1.), (7., 7.)]
    def anisodiff(self, niter=12, kappa=11, gamma=0.9, step=(5.,5.)):

        self.img = self.img.astype(float)
        deltaS = np.zeros_like(self.img)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(self.img)
        gE = gS.copy()

        for ii in range(niter):

            # calculate the diffs (change in pixels in both rows and cols)
            deltaS[:-1, :] = np.diff(self.img, axis=0)
            deltaE[:, :-1] = np.diff(self.img, axis=1)

            # conduction gradients
            gS = 1. / (1. + (deltaS/kappa)**2.) / step[0]
            gE = 1. / (1. + (deltaE/kappa)**2.) / step[1]

            # update matrices
            S = gS * deltaS
            E = gE * deltaE

            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't as questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:, :] -= S[:-1, :]
            EW[:, 1:] -= E[:, :-1]

            self.img += gamma * (NS + EW)

    def threshold(self, threshold, x, y):

        # prepare img arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        self.img = np.where(self.img < threshold, x, y)
        self.round()

    def to_mask(self, polygons=None, x=0, y=1):
        if polygons is None:
            polygons = [copy(d.polygon) for d in self.get_bodies()]
        polys = []
        for p in polygons:
            if p.is_micron:
                self.converter.to_pixels(p, self.lvl)
            else:
                self.converter.rescale(p, self.lvl)
            p = np.rint(p).astype(np.int)
            polys.append([tuple(p[i]) for i in range(p.shape[0])])
        mask = Image.new('L', (self.size()[0], self.size()[1]), x)
        for poly in polys:
            ImageDraw.Draw(mask).polygon(poly, outline=y, fill=y)
        self.img = np.array(mask)[:, :, None]

    def get_tissue_threshold(self, blur=0, mi=0, mx=255):
        self.gauss_blur(blur)
        data = self.img.flatten()
        data = data[data >= mi]
        data = data[data <= mx]
        data = data[:, None]
        return sana_thresholds.kittler(data)[0]

    def get_stain_threshold(self, tissue_mask=None, blur=0, mi=0, mx=255):
        self.gauss_blur(blur)
        if not tissue_mask is None:
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
        layer_0_rot = layer_0.rotate(roi.centroid()[0], angle)
        if np.mean(layer_0_rot[:, 1]) >= self.size()[1]//2:
            angle -= 180
        return angle

    def detect(self, tree=True):
        self.detections = []
        if tree:
            retr = cv2.RETR_TREE
        else:
            retr = cv2.RETR_EXTERNAL
        contours, hierarchies = cv2.findContours(
            self.img, retr, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchies is None:
            return

        for c, h in zip(contours, hierarchies[0]):
            self.detections.append(Detection(c, h, self.lvl, self.converter))

    # filter the detections into groups based on the area and the hierarchy
    def filter(self, min_body_area=0, min_hole_area=0, max_body_area=None, max_hole_area=None):

        # find the body detections
        # NOTE: these are detections without a parent and above the min size
        for d in self.detections:
            if d.hierarchy[3] != -1:
                continue
            if d.polygon.area() <= min_body_area:
                continue
            if not max_body_area is None and d.polygon.area() >= max_body_area:
                continue
            d.body = True

        # find the holes in the body detections
        for d in self.detections:
            if d.body:
                continue
            if not self.detections[d.hierarchy[3]].body:
                continue
            if d.polygon.area() <= min_hole_area:
                continue
            if not max_hole_area is None and d.polygon.area() >= max_hole_area:
                continue
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

    def get_cells(self, x0, y0, x1, y1):
        cells = []
        for d in self.detections:
            if not d.body:
                continue
            p = d.polygon
            c = p.centroid()[0]
            if c[0] >= x0 and c[0] <= x1 and c[1] >= y0 and c[1] <= y1:
                cells.append(d)
        return cells

    def detect_tissue(self, threshold, min_body_area=0, min_hole_area=0):

        # detect the objects on the thresholded image
        self.to_gray()
        self.gauss_blur(1)
        self.threshold(threshold, x=255, y=0)
        self.detect()

        # filter the detections based on the given areas
        self.filter(min_body_area=min_body_area, min_hole_area=min_hole_area)

    def detect_cells(self, min_body=0, max_body=None):
        self.detect(tree=False)
        self.filter(min_body_area=min_body, max_body_area=max_body)

    def detect_cell_centers(self, radius, gap):

        # apply distance transform to find centers of cells
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dist = cv2.distanceTransform(
            self.img, cv2.DIST_HUBER, cv2.DIST_MASK_PRECISE)
        dist = cv2.copyMakeBorder(
            dist, radius, radius, radius, radius,
            cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        dist = cv2.erode(dist, kern)

        # apply template matching with a circle template
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*(radius-gap)+1, 2*(radius-gap)+1))
        kern = cv2.copyMakeBorder(
            kern, gap, gap, gap, gap,
            cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        temp = cv2.distanceTransform(
            kern, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        corr = cv2.matchTemplate(dist, temp, cv2.TM_CCOEFF)

        # threshold the peaks of the template matching
        mn, mx, _, _ = cv2.minMaxLoc(corr)
        th, peaks = cv2.threshold(corr, mx*0.1, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)

        # calculate the center of each thresholding peak
        centers = []
        contours, hierarchy = cv2.findContours(
            peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # _, mx, _, mxloc = cv2.minMaxLoc(
            #     self.img[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
            # centers.append(np.array([mxloc[0]+x, mxloc[1]+y]))
            centers.append(np.array((x+w//2, y+h//2)))
        # fig, axs = plt.subplots(1,3)
        # axs[0].imshow(self.img)
        # axs[1].imshow(dist)
        # axs[2].imshow(peaks)
        # for c in centers:
        #     axs[0].plot(c[0], c[1], '.', color='red')
        #     axs[2].plot(c[0], c[1], '.', color='red')
        # plt.show()

        return centers

    def detect_cell_sizes(self, centers, plot=False):
        i = 2*self.img[:, :, 0].astype(np.int) - 255
        mi = 3
        mx = 39
        gap = 5
        rs = list(range(mi, mx+1, 2))
        temps = []
        sizes = []
        for r in rs:
            temp = np.full((2*mx+1+gap+gap, 2*mx+1+gap+gap), 0)
            c = temp.shape[0]//2

            kern = 2 * cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            a, b = c - kern.shape[0]//2, c + kern.shape[0]//2 + 1
            temp[a:b, a:b] += kern

            kern = -1 * cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*r+gap,2*r+gap))
            a, b = c - kern.shape[0]//2, c + kern.shape[0]//2 + 1
            temp[a:b, a:b] += kern

            temps.append(temp)

        for c in centers:
            a, b = c - temp.shape[0]//2, c + temp.shape[0]//2 + 1
            x0pad, x1pad, y0pad, y1pad = 0, 0, 0, 0
            if a[0] < 0:
                x0pad = 0 - a[0]
                a[0] = 0
            if a[1] < 0:
                y0pad = 0 - a[1]
                a[1] = 0
            if b[0] > i.shape[1]:
                x1pad = b[0] - i.shape[1]
                b[0] = i.shape[1]
            if b[1] > i.shape[0]:
                y1pad = b[1] - i.shape[0]
                b[1] = i.shape[0]

            x = i[a[1]:b[1], a[0]:b[0]]
            y0pad = np.full_like(x, -255, shape=(y0pad, x.shape[1]))
            y1pad = np.full_like(x, -255, shape=(y1pad, x.shape[1]))
            x = np.concatenate((y0pad, x, y1pad), axis=0)
            x0pad = np.full_like(x, -255, shape=(x.shape[0], x0pad))
            x1pad = np.full_like(x, -255, shape=(x.shape[0], x1pad))
            x = np.concatenate((x0pad, x, x1pad), axis=1)
            if x.shape != temp.shape:
                print(x.shape, temp.shape, c)
                sizes.append(0)
                continue
            corrs = []
            for r, temp in zip(rs, temps):
                corr = x * temp
                corrs.append(np.sum(corr) / r)
                if plot:
                    print(corrs[-1], flush=True)
                    fig, axs = plt.subplots(1,3)
                    axs[0].imshow(x)
                    axs[1].imshow(temp)
                    axs[2].imshow(corr)
                    plt.show()
            inc = [corrs[i+1] - corrs[i] for i in range(len(corrs)-1)]
            ind = [i for i, x in enumerate(inc) if x<0]
            if len(ind) == 0:
                ind = len(corrs)-1
            else:
                ind = ind[0]
            sizes.append(rs[ind])
        return sizes
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
