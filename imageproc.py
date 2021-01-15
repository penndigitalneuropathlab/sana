
# TODO: should probably remove as much PIL as possible, switching from
#        Image to array and vice versa can't be efficient
import os
import cv2
import numpy as np
from numpy.linalg import det, inv
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import rescale_intensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# applies a Gaussian Blur to a PIL Image object
# TODO: radius should be in terms of an actual neuropath value
def gauss_blur(img, sigma):
    return gaussian_filter(img, sigma=sigma).astype(np.uint8)

def simple_threshold(img, threshold, x=0, y=1):
    x = np.full_like(img, x)
    y = np.full_like(img, y)
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
    return contours, hier

def upscale_contours(contours, ds):
    for i in range(len(contours)):
        contours[i] = contours[i].astype(np.float64) * ds
    return contours
def downscale_contours(contours, ds):
    for i in range(len(contours)):
        contours[i] = contours[i].astype(np.float64) / ds
    return contours
def round_contours(contours):
    return np.rint(contours).astype(np.int32)

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

def get_neuron_density(loader, frame, tile_size, tile_step):

    # generate the tiles from the frame
    tiles = loader.load_tiles(frame, tile_size, tile_step)

    # TODO: incorporate avg neuron size to get number of neurons in tile instead of density of neurons
    # calculate the neuron density of each tile
    # NOTE: https://www.kenhub.com/en/library/anatomy/cortical-cytoarchitecture
    return 1 - (np.sum(tiles, axis=(2, 3)) / (255 * tile_size[0]*tile_size[1]))

# finds the tissue in a thresholded image
def detect_tissue(loader, img, lvl):

    # convert to grayscale and blur
    gray = rgb_to_gray(img)
    blur = gauss_blur(gray, 7)

    # TODO: detect from histo
    # apply the threshold
    threshold = 230
    thresh = simple_threshold(blur, threshold, x=1, y=0)

    # get the contours of the image
    contours, hierarchy = find_contours(loader, thresh, lvl)

    # find all the main tissue specimens
    # NOTE: contours without a parent and is above the min tissue size
    # TODO: get an actual good micron^2 value
    min_tissue_area = 1e5
    for c in contours:
        if c[1][3] == -1 and c[2] > min_tissue_area:
            c[3] = 1

    # find all the holes in the main tissue specimens
    # NOTE: contours whose parent is a main tissue specimen and is
    #       above the min hole size
    # TODO: get an actual good micron^2 value
    min_hole_area = 1e6
    for c in contours:
        if contours[c[1][3]][3] == 1 and c[2] > min_hole_area:
            c[3] = -1

    # TODO: currently ignoring all other contours in the hierarchy
    pass

    # separate out the main tissue contours, and the tissue hole contours
    # NOTE: ignoring all other contours (these should be background)
    tissue_contours = [c[0] for c in contours if c[3] == 1]
    hole_contours = [c[0] for c in contours if c[3] == -1]

    # draw the contours on the image
    img = cv2.drawContours(img, tissue_contours, -1,
                           (0, 0, 0), thickness=3)
    img = cv2.drawContours(img, hole_contours, -1,
                           (0, 0, 0), thickness=3)

    # upscale contours to original resolution
    tissue_contours = upscale_contours(tissue_contours, loader.get_ds(lvl))

    return tissue_contours, thresh, img

def detect_wm(loader, nd, tb, tissue_contours, tile_ds):

    # downscale tissue contours to the tiled resolution
    tissue_contours = downscale_contours(
        tissue_contours, loader.get_ds() * tile_ds)
    tissue_contours = round_contours(tissue_contours)

    # mask out the background
    mask = generate_mask(tissue_contours, nd.shape)
    nd_tissue = mask_image(nd, mask)

    # get the histogram, set the background count to zero
    hist = histogram(nd_tissue)
    hist[0] = 0

    # TODO: can do this before masking and rounding!!
    # find the boundary between gm and wm
    gm_wm, means, vars = mle_threshold(nd_tissue)

    # threshold the image based the on gm/wm boundary
    # TODO: probably need a K-Means or something... balanced is so weird
    gm = np.full_like(nd_tissue, 200)
    nd_thresh = np.where(nd_tissue < gm_wm, nd_tissue, gm)
    # nd_thresh = simple_threshold(nd_tissue, gm_wm, x=1, y=2)

    # TODO: how does this work? seems like it might be doing 0 to 1+
    # TODO: need to blur and/or increase micron size
    # TODO: need to remove small annotations, fill holes
    contours, hierachy = find_contours(loader, nd_thresh)
    contours = [c[0] for c in contours]

    tb_gm = cv2.drawContours(tb, contours, -1, 0, thickness=3)

    return contours, nd_thresh, tb_gm, hist, gm_wm, means, vars

def mle_threshold(x, k=4):

    # TODO: elbow method to find the true k
    #        --probably is 4: bckg, wm, lo gm, hi gm
    # find the initial means of the distributions using KMeans
    kmeans = KMeans(n_clusters=k).fit(x.flatten()[:, None])
    means = kmeans.cluster_centers_

    # perform a GMM to find the mean and variances of our distributions
    gmm = GaussianMixture(n_components=k, means_init=means).fit(x.flatten()[:, None])
    means = gmm.means_[:, 0]
    vars = gmm.covariances_[:, 0, 0]

    # get the wm and gm distributions
    # NOTE: 2nd and 3rd means in an ordered list, 1st is the background
    inds = means.argsort()[1:3]
    means = means[inds]
    vars = vars[inds]

    # perform maximum likelihood estimation, find the crossing of the PDFs
    t = np.arange(0, 256, 1)
    p_wm = multivariate_normal(means[0], vars[0]).pdf(t).flatten()
    p_gm = multivariate_normal(means[1], vars[1]).pdf(t).flatten()
    threshold = np.nonzero(p_wm > p_gm)[0][-1]

    return threshold, np.rint(means).astype(int), np.rint(vars).astype(int)

def balanced_hist_threshold(x, i_s=0, i_e=255):

    # move start and end points until we hit relevant area of the histo
    min_color = np.mean(x[i_s:i_e])
    while x[i_s] < min_color:
        i_s += 1
    while x[i_e] < min_color:
        i_e -= 1

    # find center of histo
    i_m = int(round((i_s + i_e) / 2))

    # calculate the weights of the left and right sides of the histo
    w_l = np.sum(x[i_s:i_m])
    w_r = np.sum(x[i_m:i_e+1])

    # loop until the start and ends meet
    while i_s < i_e:

        # left side heavier, shift the start up
        if w_l > w_r:
            w_l -= x[i_s]
            i_s += 1
        # right side heavier, shift the end down
        else:
            w_r -= x[i_e]
            i_e -= 1

        # readjust the weights if the middle changed
        new_i_m = int(round((i_s + i_e) / 2))
        if new_i_m < i_m:
            w_l -= x[i_m]
            w_r += x[i_m]
        elif new_i_m > i_m:
            w_l += x[i_m]
            w_r -= x[i_m]
        i_m = new_i_m

    return i_m

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

    def run(self, frame, ret=[True, True, True], rescale=True):
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





#
# end of file
