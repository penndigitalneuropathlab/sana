
# TODO: should probably remove as much PIL as possible, switching from
#        Image to array and vice versa can't be efficient
import os
import cv2
import numpy as np
from numpy.linalg import det, inv
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import imageio as iio

# applies a Gaussian Blur to a PIL Image object
# TODO: radius should be in terms of an actual neuropath value
def gauss_blur(img, sigma=3):
    return gaussian_filter(img, sigma=sigma).astype(np.uint8)

def simple_threshold(img, threshold, lo=0, hi=1):
    x = np.full_like(img, lo)
    y = np.full_like(img, hi)
    return np.where(img < threshold, x, y)

# calculates the area of a contour (polygon)
# TODO: make sure that vertice order doesn't matter, this might be shoelace algo
def calc_contour_area(contour):
    x = np.array([float(v[0][0]) for v in contour])
    y = np.array([float(v[0][1]) for v in contour])
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# finds the contours of a thresholded image
# TODO: is chain approx simple sufficient? seems like it just saves mem
def find_contours(loader, img, lvl=None):
    cont, hier = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # combine the contours and the hierachy into a new data structure
    #  NOTE: [contour vertices, hierachy placement, micron area, status]
    #  NOTE: status is {-1: hole, 0: background, 1: tissue}
    contours = []
    for c, h in zip(cont, hier[0]):

        # get the pixel area of the contour in current pixel resolution
        px_area = calc_contour_area(c)

        # convert the pixel area to the original resolution micron area
        micron_area = loader.px_to_micron(px_area, lvl, order=2)

        # store data in data structure
        contours.append([c, h, micron_area, 0])
    return contours

def filter_contours(contours, min_main_area, min_hole_area):

    # find all the main contours
    # NOTE: contours without a parent and is above the min size
    for c in contours:
        if c[1][3] == -1 and c[2] > min_main_area:
            c[3] = 1

    # find all the holes in the main tissue specimens
    # NOTE: contours whose parent is a main tissue specimen and is
    #       above the min hole size
    # TODO: get an actual good micron^2 value
    for c in contours:
        if c[3] != 1 and contours[c[1][3]][3] == 1 and c[2] > min_hole_area:
            c[3] = -1

    # TODO: currently ignoring all other contours in the hierarchy
    pass

    # separate out the main contours, and the hole contours
    # NOTE: ignoring all other contours (these should be background)
    main_contours = [c[0] for c in contours if c[3] == 1]
    hole_contours = [c[0] for c in contours if c[3] == -1]

    return main_contours, hole_contours

def upscale_contours(contours, ds):
    for i in range(len(contours)):
        contours[i] = contours[i].astype(np.float64) * ds
    return contours
def downscale_contours(contours, ds):
    for i in range(len(contours)):
        contours[i] = contours[i].astype(np.float64) / ds
    return contours
def round_contours(contours):
    for i in range(len(contours)):
        contours[i] = np.rint(contours[i]).astype(np.int32)
    return contours

# converts the detections into the contour using the pixel resolution and the relative position of the tile
def detections_to_contours(loader, detections, loc=None):
    if loc is None:
        loc = np.array(0,0)

    return [loader.micron_to_px(
        np.reshape(d, (d.shape[0]//2, 1, 2))) - loc for d in detections]

def rgb_to_gray(img):
    return np.dot(img.astype(np.float64), [0.2989, 0.5870, 0.1140])

def histogram(img):
    return np.histogram(img, bins=256, range=(0, 255))[0]

def round_img(img):
    return np.rint(img).astype(np.uint8)

def norm(v):
    k = (np.sqrt(np.sum(v**2)))
    if k != 0:
        return v / k
    else:
        return v

# creates an image mask based on the given detections
def generate_mask(contours, size):

    # draw the contours, inside detections stays white, outside turns black
    return cv2.drawContours(np.full(size, 0, dtype=np.uint8),
                            contours, -1, 255, -1)

# applies the mask to the image, non-mask areas turn black
def mask_image(img, mask):
    return cv2.bitwise_and(img, mask)

def get_slide_color(img):
    img = rgb_to_gray(img)
    img = round_img(img)
    histo = histogram(img)
    return np.tile(np.argmax(histo), 3)

def get_neuron_density(tiles):

    # TODO: incorporate avg neuron size to get number of neurons in tile instead of density of neurons
    # calculate the neuron density of each tile
    # NOTE: https://www.kenhub.com/en/library/anatomy/cortical-cytoarchitecture
    return round_img(255 - np.mean(tiles, axis=(2, 3)))


def gmm(x, k):

     # perform a GMM to find the mean and variances of our distributions
    gmm = GaussianMixture(n_components=k, covariance_type='full') \
        .fit(x.flatten()[:, None])

    # get the sorted means and corresponding variances
    inds = gmm.means_[:, 0].argsort()
    means = gmm.means_[:, 0][inds]
    vars = gmm.covariances_[:, 0, 0][inds]

    return np.rint(means).astype(int), vars


def mle(means, vars):

    # perform maximum likelihood estimation
    # NOTE: this is finding the crossing of the PDFs between the means
    t = np.arange(0, 256, 1)
    thresholds = []
    for i in range(means.shape[0]-1):

        # generate the pdfs
        p1 = multivariate_normal(means[i], vars[i]).pdf(t)
        p2 = multivariate_normal(means[i+1], vars[i+1]).pdf(t)

        # only evaluate the pdfs between the 2 means
        p1 = p1.flatten()[means[i]:means[i+1]]
        p2 = p2.flatten()[means[i]:means[i+1]]

        # find the last x value where p1 is more probable than p2
        thresholds.append(np.nonzero(p1 > p2)[0][-1] + means[i])
    return thresholds

def detect_wm(loader, nd, tb, tile_ds):

    # get the histogram, scale to size of the image
    hist = histogram(nd) / (nd.shape[0] * nd.shape[1])

    # segment the image based on the distribution of colors
    means, vars = gmm(nd, 3)
    bg_wm, wm_gm = mle(means, vars)

    # threshold the image for the wm region
    gm = np.full_like(nd, means[2])
    thresh = np.where(nd <= bg_wm, np.zeros_like(nd), np.ones_like(nd))
    thresh = np.where(nd > wm_gm, np.zeros_like(nd), thresh)

    # blur and re-threshold to remove thin strips
    thresh = gauss_blur(255*thresh, 7)
    thresh = np.where(thresh>128, np.ones_like(thresh), np.zeros_like(thresh))

    # find the contours of the wm region
    contours = find_contours(loader, thresh.astype(np.uint8))

    wm_contours, hole_contours = filter_contours(contours, 1e4, 1e5)
    wm_contours = upscale_contours(wm_contours, tile_ds)
    wm_contours = round_contours(wm_contours)

    cv2.drawContours(tb, wm_contours, -1, 0, thickness=3)

    return contours, thresh, hist, wm_gm, means, vars

# TODO: estimate the stain vector, dont just use the default
# NOTE: does not support removing dyes yet, ask Claire what this is
# [0] code from skimage.color
# [1] A Model based Survey of Colour Deconvolution in Diagnostic Brightfield Microscopy: Error Estimation and Spectral Consideration
class StainSeparator:
    def __init__(self, stain_type):

        # define the staining vector
        # NOTE: NeuN uses H-DAB, HED also works since we only care about H and DAB
        if stain_type == 'H-DAB':
            stain_v = np.array([
                [0.65, 0.70, 0.29],
                [0.27, 0.57, 0.78],
                [0.00, 0.00, 0.00],
            ])
        elif stain_type == 'HED':
            stain_v = np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78]
            ])

        # normalize the vectors
        # NOTE: usually the stain vector will already be normalized
        stain_v[0, :] = norm(stain_v[0, :])
        stain_v[1, :] = norm(stain_v[1, :])
        stain_v[2, :] = norm(stain_v[2, :])

        # 3 color is unspecified, create an orthogonal residual color
        # NOTE: this color will sometimes have negative components, thats okay
        #        since we will check for this later on
        if all(stain_v[2, :] == 0):
            stain_v[2, :] = np.cross(stain_v[0, :], stain_v[1, :])

        # generate the inverse for converting rgb to stain separation
        stain_v_inv = inv(stain_v)

        # store the vectors convert rgb to stain and vice versa
        self.stain_to_rgb = stain_v
        self.rgb_to_stain = stain_v_inv

    def dab_gray(self, frame):
        _, dab, _ = self.run(frame, ret=(False, True, False), rescale=False)
        return rgb_to_gray(dab)

    def run(self, frame, ret=[True, True, True], rescale=False):
        s1, s2, s3 = None, None, None

        # separate the stains, convert from rgb to stains
        stains = self.separate_stains(frame)

        # convert each stain channel individually back to a new rgb image
        #  using the original stain vector
        z = np.zeros_like(stains[:, :, 0])
        if ret[0]:
            s1 = self.combine_stains(
                np.stack((stains[:, :, 0], z, z), axis=-1))
        if ret[1]:
            s2 = self.combine_stains(
                np.stack((z, stains[:, :, 1], z), axis=-1))
        if ret[2]:
            s3 = self.combine_stains(
                np.stack((z, z, stains[:, :, 2]), axis=-1))

        if not rescale:
            return s1, s2, s3
        else:
            h = rescale_intensity(stains[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 0], 99)))
            d = rescale_intensity(stains[:, :, 1], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 1], 99)))
            return s1, s2, s3, np.dstack((z, d, h))

    # performs color denconvolution using a rgb image and a inverted stain vector
    def separate_stains(self, rgb):
        v = self.rgb_to_stain

        # scale rgb image from 0 to 1
        rgb = rgb.astype(np.float64) / 255

        # handle log errors
        np.maximum(rgb, 1e-6, out=rgb)
        log_adjust = np.log(1e-6)

        # calculate the optical density, then multiply the inverted stain vector
        stains = (np.log(rgb) / log_adjust) @ v

        # make sure there are no negative stain values
        return np.maximum(stains, 0)

    # converts the stain separated image back to an rgb image using the stain vector
    def combine_stains(self, stain):
        v = self.stain_to_rgb

        # handle log errors
        log_adjust = -np.log(1e-6)

        # generate the rgb image using the stain intensity and the stain vector
        rgb = np.exp(-(stain * log_adjust) @ v)

        # clip values from 0 to 255, then convert to 8 bit ints
        return np.rint(255 * np.clip(rgb, a_min=0, a_max=1)).astype(np.uint8)
    #
    # end of class

def run_neuron_density(f, lvl, fsize, fstep, tsize, tstep):

    # initialize the objects
    separator = StainSeparator('H-DAB')
    loader = iio.SVSLoader(f)
    loader.set_lvl(lvl)
    loader.set_frame_dims(fsize, fstep)
    loader.set_tile_dims(tsize, tstep)

    # define the framing and tiling functions and arguments
    ffunc = separator.dab_gray
    fargs = []

    tfunc = get_neuron_density
    targs = []

    # run the framing and tiling
    ndf = loader.run_tiling(ffunc, fargs, tfunc, targs)

    # stitch the frame analyses together
    cols = []
    for j in range(ndf.shape[1]):
        cols.append(np.concatenate(ndf[:, j], axis=1))
    nd = np.concatenate(cols, axis=0)

    # crop out the padded portions of the frames
    w, h = np.rint(loader.get_dim() / loader.tds).astype(np.int)
    return nd[:h, :w]

# TODO: need a way to remove area if majority is white, see airbubbles and external tissue fragments
def run_tissue_detection(f):

    loader = iio.SVSLoader(f)

    # convert thumbnail to grayscale and blur
    gray = rgb_to_gray(loader.thumbnail)
    histo = histogram(round_img(gray))
    blur = gauss_blur(gray, 5)

    # define the threshold as just above the mean tissue color
    # NOTE: this removes the external tissue surrounding the cortical ribbon
    m, v = gmm(blur, 2)
    threshold = int(m[0] + 2*np.sqrt(v[0]))

    # TODO: use threhsold to the left to remove airbubbles and other dark object
    # apply the threshold based on the color of the slide
    # threshold = int(np.rint(rgb_to_gray(loader.slide_color))) - 10
    thresh = simple_threshold(blur, threshold, lo=1, hi=0)

    # get the contours of the thresholded image
    contours = find_contours(loader, thresh, loader.get_thumbnail_lvl())

    # remove unnecessary detections
    tissue_contours, hole_contours = filter_contours(contours, 1e6, 1e6)

    # draw the contours on the thumbnail
    cv2.drawContours(loader.thumbnail, tissue_contours, -1,
                     (255, 0, 0), thickness=3)
    cv2.drawContours(loader.thumbnail, hole_contours, -1,
                     (255, 0, 0), thickness=3)

    # upscale contours to original resolution
    # NOTE: currently ignoring the holes, need to subtract these from the tissue
    tissue_contours = upscale_contours(
        tissue_contours, loader.get_ds(loader.get_thumbnail_lvl()))

    return tissue_contours, thresh, loader.thumbnail, histo, threshold
#
# end of file
