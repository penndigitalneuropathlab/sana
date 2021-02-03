
import os
import sys
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from numba import jit
from scipy import ndimage

import imageio as iio
import imageproc as iproc

# NOTE: https://link.springer.com/article/10.1007/s10803-009-0790-8
# this paper is basically donig what im doing from 2009

F_SRC = \
"~/neuropath/data/layer_segmentation/images/2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs"

def main(argv):

    # parse the command line
    if '-f' not in argv:
        f = F_SRC
    else:
        f = argv[argv.index('-f')+1]
    f = iio.get_fullpath(f)
    a = f.replace('images', 'polygons').replace('.svs', '.svs.json')

    # initialize the SVS loader
    lvl = 2
    loader = iio.SVSLoader(f)
    loader.set_lvl(lvl)

    # get a mask defining the location of background and foreground
    tissue_mask = iproc.get_bg_fg_mask(np.copy(loader.thumbnail))

    # load the annotations
    annos = iio.read_qupath_annotations(a)
    for i, anno in enumerate(annos):
        if i == 0:
            continue

        # downscale the annotation to the current pixel resolution
        anno = loader.downscale_px(anno)

        # calculate the centroid of the annotation
        centroid = iproc.calc_centroid(anno[:, 0], anno[:, 1])

        # find the max distance from the centroid
        dist = np.sqrt((centroid[0]-anno[:, 0])**2 + (centroid[1]-anno[:, 1])**2)
        ri = np.argmax(dist)
        r = np.max(dist)

        nds = []
        sigs = []

        tsize = np.array((75, 75))
        tstep = np.array((10, 10))
        loader.set_tile_dims(tsize, tstep)
        nd, sig = run_layer_segmentation(loader, tissue_mask, anno, centroid, r)
        nds.append(nd)
        sigs.append(sig)

        tsize = np.array((10, 100))
        tstep = np.array((10, 10))
        loader.set_tile_dims(tsize, tstep)
        nd, sig = run_layer_segmentation(loader, tissue_mask, anno, centroid, r)
        nds.append(nd)
        sigs.append(sig)

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        for i in range(2):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(nds[i].T, cmap='coolwarm')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_xlim([0, nds[i].T.shape[1]])
            ax.set_ylim([nds[i].T.shape[0], 0])
        for i in range(2):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(sigs[i], color='black')
            ax.set_xlim([0, sigs[i].shape[0]])
        plt.show()

def run_layer_segmentation(loader, tissue_mask, anno, centroid, r):

    # define the frame size and location based on the centroid and radius
    floc = np.rint(centroid - r).astype(np.int)
    fsize = np.rint(np.array([2*r, 2*r])).astype(np.int)

    # shift the frame loc to account for center aligned tiles
    floc -= loader.tsize//2
    fsize += loader.fpad

    # load the frame region
    frame = loader.load_region(floc, fsize)

    # shift mask to the relative position
    mask = tissue_mask[floc[1]:floc[1]+fsize[1], floc[0]:floc[0]+fsize[0]]

    # shift the annotation to relative position and account for
    #  center aligned tiles
    anno = np.copy(anno) - floc
    centroid = np.copy(centroid) - floc

    # find the contour of the pial boundary
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # get the vertices of the contours within the annotation
    layer0 = []
    for c in contours:
        for v in c:
            if ray_tracing(v[0][0], v[0][1], anno):
                layer0.append(v[0])
    layer0 = np.array(layer0)

    # do linear regression to find the best fit for the pial boundary
    m0, b0 = linear_regression(layer0[:, 0], layer0[:, 1])
    tissx = np.array([0, fsize[0]])
    tissy = m0*tissx + b0

    # find the rotation of the best fit
    angle = iproc.find_angle((tissx[0], tissy[0]), (tissx[1], tissy[1]))
    angle = angle - (90 * (angle//90))

    # NOTE: reshape=False might cause alignment issue down the line
    # rotate the frame by the angle of rotation
    frame_rot = ndimage.rotate(frame, angle, reshape=False, mode='nearest')

    # rotate the annotation by the angle of rotation
    th = math.radians(-angle)
    anno_rot = np.array([np.rint(
        iproc.rotate_point(centroid, anno[i], th)).astype(int) \
                         for i in range(anno.shape[0])])


    separator = iproc.StainSeparator('H-DAB')
    dab = separator.dab_gray(frame_rot)

    tiles = loader.load_tiles(dab)

    nd = iproc.get_neuron_density(tiles)
    # nd = nd[:, 200:250]
    sig = np.mean(nd, axis=1)

    return nd, sig

def plot_rotation(frame, anno, centroid, r, mask, tissx, tissy, frame_rot, anno_rot):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # plot the original image, with the geometry of the polygon and rotation
    axs = []
    ax = fig.add_subplot(gs[0, 0])
    axs.append(ax)
    ax.imshow(frame)
    plot_vertices(ax, anno)
    ax.plot(centroid[0], centroid[1], '.')
    circle = plt.Circle(centroid, r, color='black', fill=False)
    ax.add_patch(circle)

    # plot the tissue mask, with the detected line of best fit
    ax = fig.add_subplot(gs[0, 1])
    axs.append(ax)
    ax.imshow(mask)
    ax.plot(tissx, tissy, color='red')

    # plot the rotated image and annotation
    ax = fig.add_subplot(gs[0, 2])
    axs.append(ax)
    ax.imshow(frame_rot)
    plot_vertices(ax, anno_rot)

    for ax in axs:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlim([0, frame.shape[1]])
        ax.set_ylim([frame.shape[0], 0])
    plt.show()

def linear_regression(x, y):
    n = x.shape[0]
    ss_xy = np.sum(y * x) - n * np.mean(y) * np.mean(x)
    ss_xx = np.sum(x**2) - n * np.mean(x)**2
    m = ss_xy/ss_xx
    b = np.mean(y) - m * np.mean(x)
    return m, b

# @jit(nopython=True)
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

def plot_vertices(ax, v):
    for i in range(v.shape[0] - 1):
        ax.plot([v[i][0], v[i+1][0]], [v[i][1], v[i+1][1]], color='red')


if __name__ == "__main__":
    main(sys.argv)
