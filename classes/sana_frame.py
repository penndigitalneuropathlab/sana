
# system packages
import os
import sys
from copy import copy

# installed packages
import cv2
import numpy as np
import nibabel as nib
from scipy import ndimage
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from matplotlib import pyplot as plt
from webcolors import name_to_rgb

# custom packages
from sana_geo import Point, Polygon, Line, ray_tracing, separate_seg, plot_poly
from sana_color_deconvolution import StainSeparator
from sana_thresholds import kittler
from sana_tiler import Tiler

# custom exceptions
class TypeException(Exception):
    def __init__(self, message):
        self.message = message

# TODO: should add flag to store whether image is binary
#        or detect somehow? integer and only 2 values (0 and 1)
# provides a series of functions to apply to a 2D array stored in it's memory
#  -img: 2D Numpy array
#  -lvl: pixel resolution of img
#  -converter: Converter object from Loader
class Frame:
    def __init__(self, img, lvl=-1, converter=None,
                 csf_threshold=None, slide_color=None, padding=0):
        if type(img) is str:
            self.img = np.array(Image.open(img))
        else:
            self.img = img

        # make sure the img always has a pixel channel
        if self.img.ndim < 3:
            self.img = self.img[:, :, None]
        self.lvl = lvl
        self.converter = converter
        self.contours = []
        self.csf_threshold = csf_threshold
        self.slide_color = slide_color
        self.padding = padding

    #
    # end of constructor

    # checks if the image array has 3 channels
    def is_rgb(self):
        if len(self.img.shape) == 2:
            return False
        else:
            return self.img.shape[2] == 3
    #
    # end of is_rgb

    # checks to see if the image array is a 1 channel, 1 byte, 0 to 1 image
    def is_binary(self):
        if self.is_rgb():
            return False
        if self.img.dtype != np.uint8:
            return False
        if np.max(self.img) > 1:
            return False
        return True
    #
    # end of is_binary

    # checks if the image array is a float or a 1 byte image
    def is_float(self):
        if self.img.dtype != np.uint8:
            return True
    #
    # end of is_float

    # gets the size of the image
    def size(self):
        size = Point(self.img.shape[1], self.img.shape[0], False, self.lvl)
        return self.converter.to_int(size)
    #
    # end of size

    # returns a duplicate of the Frame
    # NOTE: this is useful because most functions do not create copies
    def copy(self):
        return Frame(np.copy(self.img), self.lvl, self.converter, self.csf_threshold)
    #
    # end of copy

    # converts an RGB image to a grayscale image
    def to_gray(self):
        if not self.is_rgb():
            return
        float_img = self.img.astype(np.float64)
        self.img = np.dot(float_img, [0.2989, 0.5870, 0.1140])[:, :, None]
    #
    # end of to_gray

    # repeats a 1-channel image to create a 3-channel image
    def to_rgb(self):
        if self.is_rgb():
            return
        self.img = np.tile(self.img, 3)
    #
    # end of to_rgb

    # rounds the image to a 1 byte int image
    def round(self):
        self.img = np.rint(self.img).astype(np.uint8)
    #
    # end of round

    def rescale(self, mi=None, mx=None):
        if mi is None:
            mi = np.min(self.img)
        if mx is None:
            mx = np.max(self.img)

        self.img = 255 * (self.img.astype(float) - mi) / (mx - mi)
        self.round()

    # generates a 256 bin histogram for each color channel
    def histogram(self):
        histogram = np.zeros((256, self.img.shape[-1]))
        for i in range(histogram.shape[-1]):
            histogram[:, i] = np.histogram(self.img[:, :, i],
                                           bins=256, range=(0, 255))[0]
        return histogram
    #
    # end of histogram

    # returns a rectangle extracted from the frame
    def get_tile(self, loc, size):
        self.converter.to_pixels(loc, self.lvl)
        self.converter.to_pixels(size, self.lvl)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        if loc[0] < 0: # TODO: sometimes bounding_box gives us negative values
            loc[0] = 0
        if loc[1] < 0:
            loc[1] = 0
        return self.img[loc[1]:loc[1]+size[1], loc[0]:loc[0]+size[0]]
    #
    # end of get_tile

    def get_intensity(self, poly):
        loc, size = poly.bounding_box()
        tile = self.get_tile(loc, size)
        p = poly.copy()
        p.translate(loc)
        mask = create_mask([p], size, self.lvl, self.converter)
        tile[mask == 0] = 0

        return np.mean(tile)
    #
    # end of get_intensity

    # crops the image array to a new size using a loc/size variables
    def crop(self, loc, size):
        self.img = self.get_tile(loc, size)
    #
    # end of crop

    # scales an array to a new size given a scaling factor
    def scale(self, ds, interpolation=cv2.INTER_CUBIC, size=None):
        if size is None:
            size = self.size().astype(float) / ds
            if ds[0] > 1:
                size = np.ceil(size)
            size = self.converter.to_int(size)
        self.img = cv2.resize(self.img, dsize=(size[0], size[1]),
                              interpolation=interpolation)
        if len(self.img.shape) == 2:
            self.img = self.img[:, :, None]
    #
    # end of resize

    def get_rotation_mat(self, angle):

        # get the dims and the center of the image
        h, w = self.img.shape[:2]
        cx, cy = w//2, h//2

        # get the rotation matrix and the rotation components of the mat
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])

        # compute the new bounding dims
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # translate the rotation mat to the new bounds
        M[0,2] += (nw / 2) - cx
        M[1,2] += (nh / 2) - cy

        return M, nw, nh

    def warp_affine(self, M, nw, nh, inverse=False):
        if self.is_rgb():
            border_val = (255,255,255)
        else:
            border_val = 0
            if inverse:
                M = cv2.invertAffineTransform(M)
        self.img = cv2.warpAffine(self.img, M, (nw, nh), borderValue=border_val)
        if self.img.ndim == 2:
            self.img = self.img[:,:,None]

    # rotates the image array around it's center based on a angle in degrees
    def rotate(self, angle):
        M, nw, nh = self.get_rotation_mat(angle)

        # perform the rotation
        self.warp_affine(M, nw, nh)
    #
    # end of rotate

    # pads the image array on either edge based on alignment argument
    # NOTE: changing the mode affects the GM detection
    #        - 'wrap' causes sharp peaks at edge if the tissue isn't parallel
    #        - 'symmetric' seems to be the best, the edges are very flat
    def pad(self, pad, alignment='center'):
        if alignment == 'left':
            before = pad
            after = (0,0)
        elif alignment == 'center':
            before = pad//2
            after = pad - before
        else:
            before = (0,0)
            after = pad
        self.img = np.pad(self.img,
                          ((before[1], after[1]),
                           (before[0], after[0]), (0,0)), mode='symmetric')
    #
    # end of pad

    # writes the image array to a file
    # NOTE: if the array is floating point, the image will be written as a
    #        numpy data file, else as whatever datatype is given
    def save(self, fname, invert_sform=False):
        if self.is_float():
            print(fname, 'asdjklfjklasdfjkl')
            np.save(fname.split('.')[0]+'.npy', self.img)
        else:
            if fname.endswith('.nii.gz'):
                print(fname)
                self.save_nifti(fname, invert_sform)
            else:
                im = self.img
                if not self.is_rgb():
                    im = im[:, :, 0]
                if self.is_binary():
                    im = 255 * im
                im = Image.fromarray(im)
                im.save(fname)
    #
    # end of save

    def save_nifti(self, fname, invert_sform=False):
        pix = self.img.astype(np.uint8)
        pix = np.expand_dims(pix, (2))
        spacing = [self.converter.ds[self.lvl] * (self.converter.mpp / 1000) for d in (0,1)]
        if pix.shape[-1] == 3:
            rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            pix = pix.copy().view(dtype=rgb_dtype).reshape(pix.shape[0:3])
        else:
            pix = pix[:,:,:,0]
        nii = nib.Nifti1Image(pix, np.diag([spacing[0], spacing[1], 1.0, 1.0]))
        if invert_sform:
            nii.set_sform([
                [-spacing[0], 0, 0, 0],
                [0, -spacing[1], 0, 0],
                [0, 0, 1, 0],
            ])
        nib.save(nii, fname)
    #
    # end of save_nifti

    # calculates the background color as the most common color
    #  in the grayscale space
    def get_bg_color(self):
        self.to_gray()
        histo = self.histogram()
        return np.tile(np.argmax(histo), 3)
    #
    # end of get_bg_color

    # apply a binary mask to the image
    def mask(self, mask, value=0):
        # if self.is_float():
        #     raise TypeException('Cannot apply mask to floating point image')
        if self.is_rgb() and value is int:
            value = (value, value, value)
        self.img[mask.img[:,:,0] == 0] = value
    #
    # end of mask

    # TODO: sigma should be probably be in terms of microns
    def gauss_blur(self, sigma):
        if sigma != 0:
            self.img = cv2.GaussianBlur(self.img, (sigma, sigma), 0)
        if len(self.img.shape) == 2:
            self.img = self.img[:, :, None]
    #
    # end of gauss_blur

    # TODO: need to check on these parameters and analyze the effects
    # niter - number of iterations [1, 12]
    # kappa - conduction coefficient (20-100) [50, 11]
    # gamma - max value of .25 for stability [.1, 0.9]
    # step  - distance between adjacent pixels [(1.,1.), (7., 7.)]
    def anisodiff(self, niter=12, kappa=11, gamma=0.9, step=(5.,5.)):
        if self.is_rgb():
            raise RBGException('Cannot apply filter to RGB image')
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

        # clip values between 0 and 255, round to integer
        np.clip(self.img, 0, 255, out=self.img)
        self.round()
    #
    # end of anisodiff

    # thresholds the image array based to x if below, y if above threshold
    # NOTE: x and y can be images, e.g. x = 0, y = self.img
    def threshold(self, threshold, x, y):
        if self.is_rgb():
            raise TypeException('Cannot apply threshold to RGB image')

        # prepare img arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        self.img = np.where(self.img < threshold, x, y)
        self.round()
    #
    # end of threshold

    # generates the contours on all edges of the image, then filters
    #  based on size of body and hole criteria
    def get_contours(self):

        # TODO: better error checking to handle watershed
        # if not self.is_binary():
        #     raise TypeException('Cannot get contours of a non-binary image')

        # generate the contours and store as Polygons
        self.contours = []
        contours, hierarchies = cv2.findContours(
            self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchies is None:
            return
        for c, h in zip(contours, hierarchies[0]):
            self.contours.append(Contour(c, h, self.lvl, self.converter))

    # filter the contours into groups based on the area and the hierarchy
    def filter_contours(self, min_body_area=0, min_hole_area=0,
                        max_body_area=None, max_hole_area=None):

        # find the bodies
        # NOTE: these are contours without a parent and above the min size
        for c in self.contours:
            if c.hierarchy[3] != -1:
                continue
            if c.polygon.area() <= min_body_area:
                continue
            if not max_body_area is None and d.polygon.area() >= max_body_area:
                continue
            c.body = True
        #
        # end of body checking

        # find the holes in the body contours
        # NOTE: contours with a parent that is a body and fits size criteria
        for c in self.contours:
            if c.body:
                continue
            if not self.contours[c.hierarchy[3]].body:
                continue
            if c.polygon.area() <= min_hole_area:
                continue
            if not max_hole_area is None and c.polygon.area() >= max_hole_area:
                continue
            c.hole = True
        #
        # end of hole checking
    #
    # end of filter_contours

    def get_body_contours(self):
        return [c for c in self.contours if c.body]
    def get_hole_contours(self):
        return [c for c in self.contours if c.hole]

    # get only the vertices of the contours within the given roi
    def ray_trace_contours(self, roi):

        # rescale roi to current lvl
        roi_lvl = roi.lvl
        self.converter.rescale(roi, self.lvl)

        # loop through all body contours
        x, y = [], []
        for c in self.get_body_contours():

            # loop through vertices of each body contour polygon
            p = c.polygon
            for i in range(p.shape[0]):

                # keep only the vertices that are within the roi
                if ray_tracing(p[i][0], p[i][1], np.array(roi)):
                    x.append(p[i][0])
                    y.append(p[i][1])

        # revert roi to its original lvl
        self.converter.rescale(roi, roi_lvl)

        # generate a Line array
        x, y = np.array(x), np.array(y)
        return Line(x, y, False, self.lvl)
    #
    # end of ray_trace_contours

    def get_tissue_contours(self, min_body_area=0, min_hole_area=0):

        # threshold the image for just tissue data
        self.to_gray()
        self.gauss_blur(5)
        self.round()
        self.threshold(self.csf_threshold, x=1, y=0)

        # get all contours of the image
        self.get_contours()

        # filter out small bodies and small holes
        self.filter_contours(min_body_area=min_body_area,
                             min_hole_area=min_hole_area)

        return self.contours

    # generates the CSF Line boundary from an RGB image
    # NOTE: this function can fail if given only slide or only tissue
    # TODO: does blurring affect the accuracy of the CSF boundary?
    def get_csf_boundary(self, roi):
        if not self.is_rgb():
            raise TypeException('Cannot get CSF Boundary of non-RGB image')

        # get the contours of the tissue within the image
        # TODO: get proper parameters for this
        self.get_tissue_contours(min_body_area=1e5, min_hole_area=1e5)

        # form a Line array based on the body contours within the roi
        csf_boundary = self.ray_trace_contours(roi)

        return csf_boundary
    #
    # end of get_csf_boundary

    # calculates the rotation of the Frame based on the linear regression of
    # the csf boundary detected within the given roi
    def get_rotation(self, roi):

        # get the csf boundary within the roi
        csf_boundary = self.get_csf_boundary(roi)

        # make sure we actually found a boundary
        if csf_boundary.shape[0] == 0:
            return None

        # calculate the angle of rotation based on the linear regression
        angle = csf_boundary.get_angle()

        # make sure the detected angle rotates the csf boundary to the
        # top half of the roi
        center = roi.centroid()[0]
        self.converter.rescale(center, self.lvl)
        csf_boundary.rotate(center, angle)
        if np.mean(csf_boundary[:, 1]) >= center[1]:
            angle -= 180

        return angle
    #
    # end of get_rotation

    def detect_cell_centers(self, radius, gap, plot=False):

        # apply distance transform to find centers of cells
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
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
        # TODO: this is too naive for what we need
        mn, mx, _, _ = cv2.minMaxLoc(corr)
        th, peaks = cv2.threshold(corr, mx*0.3, 255, cv2.THRESH_BINARY)
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
        if plot:
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(self.img)
            axs[1].imshow(dist)
            axs[2].imshow(peaks)
            for c in centers:
                axs[0].plot(c[0], c[1], '.', color='red')
                axs[2].plot(c[0], c[1], '.', color='red')
            plt.show()

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
                sizes.append(0)
                continue
            corrs = []
            prev_corr = -np.inf
            size = 0
            for r, temp in zip(rs, temps):
                corr = np.sum(x * temp / r)
                if corr > prev_corr:
                    prev_corr = corr
                    size = r
                else:
                    break
            sizes.append(size)
        return sizes
#
# end of Frame

class Contour:
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
        return polygon
#
# end of Detection

def frame_like(frame, img):
    return Frame(img, frame.lvl, frame.converter,
                 frame.csf_threshold, frame.slide_color, frame.padding)
#
# end of frame_like

# this function looks at the frames of data surrounding the segmentation boundaries
# and finds which boundary is associated with the tissue boundary
def get_tissue_orientation(frame, roi, angle, logger):
    
    # get the segmentation boundaries at top and bottom of frame
    s0, s1 = separate_seg(roi)
    if np.mean(s0[:,1]) > np.mean(s1[:,1]):
        top, bot = s1, s0
    else:
        top, bot = s0, s1
    
    if logger.plots:
        fig, ax = plt.subplots(1,1)
        ax.imshow(frame.img)
        plot_poly(ax, s0, color='red')
        plot_poly(ax, s1, color='blue')
        fig.show()

    # get the amount of tissue found near each of the boundaries
    top = Frame(frame.img[0:int(np.max(top[:,1])), :], frame.lvl, frame.converter, frame.csf_threshold)
    top.to_gray()
    top.threshold(top.csf_threshold, 1, 0)
    top_perc = np.sum(top.img)/top.img.size
    bot = Frame(frame.img[int(np.min(bot[:,1])):, :], frame.lvl, frame.converter, frame.csf_threshold)
    bot.to_gray()
    bot.threshold(bot.csf_threshold, 1, 0)
    bot_perc = np.sum(bot.img)/bot.img.size

    # TODO: get a new M based on the new angle and then use orig_frame to return the proper oriented frame, do this for the roi as well?

    # top image should have less tissue than the bottom
    return top_perc < bot_perc
#
# end of get_tissue_orientation

# generates a binary mask based on a list of given Polygons
#  -polygons: list of polygons to be filled with value y
#  -size: Point defining the size of the mask initialized with value x
#  -x, y: vals defining the negative and positive values in the mask
def create_mask(polygons, size, lvl, converter, x=0, y=1, holes=[], outlines_only=False):

    # convert Polygons to a list of tuples so that ImageDraw read them
    polys = []
    for p in polygons:
        converter.to_pixels(p, lvl)
        p = converter.to_int(p)
        polys.append([tuple(p[i]) for i in range(p.shape[0])])
    hs = []
    for h in holes:
        converter.to_pixels(h, lvl)
        h = converter.to_int(h)
        hs.append([tuple(h[i]) for i in range(h.shape[0])])
    #
    # end of Polygon conversion

    # TODO: make sure outline isn't increasing mask size
    # create a blank image, then draw the polygons onto the image
    size = converter.to_int(size)
    mask = Image.new('L', (size[0], size[1]), x)
    for poly in polys:
        if outlines_only:
            ImageDraw.Draw(mask).polygon(poly, outline=y, fill=0)
        else:
            ImageDraw.Draw(mask).polygon(poly, outline=y, fill=y)            
    for h in hs:
        ImageDraw.Draw(mask).polygon(h, outline=x, fill=x)
    return Frame(np.array(mask)[:, :, None], lvl, converter)
#
# end of create_mask

# TODO: just subtract gauss blur!!
# TODO: there are better methods to do this, ask paul in MRI e.g.
# this function subtracts background from the frame, by finding the local
# background intensity throughout the frame
def mean_normalize(orig_frame):
    frame = orig_frame.copy()

    # want to downsample the image to make it run faster, don't need full resolution
    lvl = 1
    ds = int(frame.converter.ds[lvl] / frame.converter.ds[frame.lvl])

    # prepare the 200x200um tiles, shift 10um
    tsize = Point(200, 200, True, 0)
    tstep = Point(10, 10, True, 0)
    tiler = Tiler(lvl, frame.converter, tsize, tstep)

    # scale down the original frame to detection background
    frame_ds = frame.copy()
    frame_ds.scale([ds, ds])
    tiler.set_frame(frame_ds)

    # loop through the tiles
    tiles = tiler.load_tiles()
    tds = tiler.ds
    norm = np.zeros([tiles.shape[0], tiles.shape[1]], dtype=float)
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):

            # calculate the local background in the tile
            data = tiles[i][j]
            data = data[data != 0]
            if len(data) == 0:
                mu = 0
            else:
                mu = np.mean(data)
            norm[i][j] = mu
    #
    # end of tiles loop

    # scale the background frame back to original resolution
    frame_norm = Frame(norm, frame.lvl, frame.converter)
    frame_norm.converter.to_pixels(tsize, lvl)

    frame_norm.scale(1/tds)
    frame_norm.scale([1/ds, 1/ds])
    frame_norm.img = frame_norm.img[:frame.img.shape[0], :frame.img.shape[1], :]

    # smooth the background image a bit
    frame_norm.img = cv2.GaussianBlur(frame.img, ksize=(0,0), sigmaX=tsize[0], sigmaY=tsize[1])[:,:,None]

    # finally, subtract the background
    # NOTE: we are making sure nothing goes below 0 here! this is okay to do since this should all be background
    frame.img = np.rint(frame.img.astype(np.float) - frame_norm.img)
    frame.img[frame.img < 0] = 0
    frame.img = frame.img.astype(np.uint8)

    return frame
#
# end of mean_normalize

# this function overlays a thresholded image onto the original image
def overlay_thresh(frame, thresh, alpha=0.5, color='red'):

    if type(color) is str:
        color = np.array(name_to_rgb(color))
    else:
        pass

    overlay = frame.copy()
    overlay.img[thresh.img[:,:,0] != 0] = color
    overlay.img = cv2.addWeighted(overlay.img, alpha, frame.img, 1-alpha, 0.0)

    return overlay
#
# end of overlay_thresh

# calculates the Slide/Tissue intensity threshold
# NOTE: should only be called by Frame
#  -data: a grayscale image array
#  -mi: minimum value to use from 0 to 255
#  -mx: maximum value to use from 0 to 255
def get_csf_threshold(frame):
    if not frame.is_rgb():
        raise TypeException('Cannot get CSF Threshold of a non-RGB image')

    # convert to grayscale
    frame.to_gray()

    # smooth to get a more accurate threshold
    frame.gauss_blur(5)

    # perform kittler thresholding
    return kittler(frame.histogram()[:, 0], mi=0, mx=255)[0]
#
# end of get_csf_threshold

# NOTE: a min value of 15 is set here based on qualitative analysis of DAB NeuN
# TODO: set different minimums for different stains!!!
def get_stain_threshold(frame, mi, mx):
    if frame.is_rgb():
        raise TypeException('Cannot get Stain Threshold of RGB image')

    # perform kittler thresholding
    return kittler(frame.histogram()[:, 0], mi=mi, mx=mx)[0]
#
# end of get_stain_threshold

#
# end of file
