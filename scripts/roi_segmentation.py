

import os
import sys
import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from numba import jit
from scipy import ndimage

import sana_io
import sana_geo
from sana_loader import Loader
from sana_framer import Framer

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
    f = sana_io.get_fullpath(f)
    a = f.replace('images', 'polygons').replace('.svs', '.svs.json')

    # initialize the SVS loader
    lvl = 2
    loader = Loader(f)
    loader.set_lvl(lvl)

    fsize = sana_geo.Point(2000, 2000, loader.mpp, loader.ds, is_micron=False, lvl=lvl)
    framer = Framer(loader, fsize)

    for i in range(framer.n[0]):
        for j in range(framer.n[1]):
            img = framer.get(i, j).img
            plt.imshow(img)
            plt.show()

#
#     # get the threshold for masking the tissue
#     tissue_threshold = iproc.get_tissue_threshold(loader.thumbnail)
#
#     # load the annotations
#     annos = iio.read_qupath_annotations(a)
#     for i, anno in enumerate(annos):
#         if i == 0:
#             continue
#
#         # downscale the annotation to the current pixel resolution
#         anno = loader.downscale_px(anno)
#
#         # calculate the centroid of the annotation
#         centroid = iproc.calc_centroid(anno[:, 0], anno[:, 1])
#
#         # find the max distance from the centroid
#         dist = np.sqrt((centroid[0]-anno[:, 0])**2 + (centroid[1]-anno[:, 1])**2)
#         ri = np.argmax(dist)
#         r = np.max(dist)
#
#         # perform short tile and long tile analysis to differentiate features
#         # NOTE: layer 1 does not show up in long tile analysis, however long tile
#         #       gives good performance on other layers
#         # NOTE: might be able to estimate a good tile size based on each frame
#         tsize = np.array((75, 500))
#         tstep = np.array((10, 10))
#         loader.set_tile_dims(tsize, tstep)
#         nd_short, sig_short = \
#                 run_layer_segmentation(
#                     loader, tissue_threshold, anno, centroid, r)
#         tsize = np.array((300, 500))
#         tstep = np.array((10, 10))
#         loader.set_tile_dims(tsize, tstep)
#         nd_long, sig_long = \
#                 run_layer_segmentation(
#                     loader, tissue_threshold, anno, centroid, r)
#         nds = [nd_short, nd_long]
#         sigs = [sig_short, sig_long]
#
#         fig = plt.figure()
#         gs = fig.add_gridspec(3, len(nds))
#         titles = ['Narrow Tile -- 75 x 500 Microns',
#                   'Wide Tile -- 300 x 500 Microns']
#         for i in range(len(nds)):
#             ax1 = fig.add_subplot(gs[:2, i])
#             ax1.imshow(nds[i].T, cmap='coolwarm')
#             ax1.grid(False)
#             ax1.set_xticks([])
#             ax1.set_yticks([])
#             ax1.set_aspect('equal')
#             ax1.set_xlim([0, nds[i].T.shape[1]])
#             ax1.set_ylim([nds[i].T.shape[0], 0])
#             ax1.set_title(titles[i])
#
#             ax2 = fig.add_subplot(gs[2, i])
#             ax2.plot(sigs[i], color='black')
#             ax2.set_xlim([0, sigs[i].shape[0]])
#             ax2.set_xlabel('Cortical Depth')
#             ax2.set_ylabel('Neuron Density (%AO DAB)')
#             ax2.set_xticks([])
#             ax1.get_shared_x_axes().join(ax1, ax2)
#         plt.suptitle('Cortical Neuron Density')
#         plt.show()
#
# def run_layer_segmentation(loader, tissue_threshold, anno, centroid, r):
#
#     # define the frame size and location based on the centroid and radius
#     floc = np.rint(centroid - r).astype(np.int)
#     fsize = np.rint(np.array([2*r, 2*r])).astype(np.int)
#
#     # shift the frame loc to account for center aligned tiles
#     floc -= loader.tsize//2
#     fsize += loader.fpad
#
#     # load the frame region
#     frame = loader.load_region(floc, fsize)
#
#     # generate the tissue mask using the pre-calculated threshold
#     tissue_mask = iproc.get_tissue_mask(frame, tissue_threshold)
#
#     # shift the annotation to relative position and account for
#     #  center aligned tiles
#     anno = np.copy(anno) - floc
#     centroid = np.copy(centroid) - floc
#
#     # find the contour of the pial boundary
#     contours = cv2.findContours(
#         tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#
#     # get the vertices of the contours within the annotation
#     layer0 = []
#     for c in contours:
#         for v in c:
#             if ray_tracing(v[0][0], v[0][1], anno):
#                 layer0.append(v[0])
#     layer0 = np.array(layer0)
#
#     # do linear regression to find the best fit for the pial boundary
#     m0, b0 = linear_regression(layer0[:, 0], layer0[:, 1])
#     tissx = np.array([0, fsize[0]])
#     tissy = m0*tissx + b0
#
#     # find the rotation of the best fit
#     angle = iproc.find_angle((tissx[0], tissy[0]), (tissx[1], tissy[1]))
#     angle = angle - (90 * (angle//90))
#
#     # TODO: annotation and frame are not being rotated correctly...
#     # NOTE: reshape=False might cause alignment issue down the line
#     # rotate the frame by the angle of rotation
#     frame_rot = ndimage.rotate(frame, angle, reshape=False, mode='nearest')
#
#     # rotate the annotation by the angle of rotation
#     th = math.radians(-angle)
#     anno_rot = np.array([np.rint(
#         iproc.rotate_point(centroid, anno[i], th)).astype(int) \
#                          for i in range(anno.shape[0])])
#
#     # generate the rotated tissue mask using the pre-calculated threshold
#     tissue_mask_rot = iproc.get_tissue_mask(frame_rot, tissue_threshold)
#
#     neuron_mask = iproc.get_neuron_mask(frame_rot, tissue_mask_rot, blur=0)
#
#     # generate the tiles from the neuron mask
#     tiles = loader.load_tiles(neuron_mask)
#
#     # TODO: start making a detector on this signal
#     #        a) heuristic peak detection
#     #        b) some type of template matching?
#     #        c) correlate with object detection (i.e. size/eccentricity)
#     nd = iproc.get_neuron_density(tiles)
#
#     # downscale the annotation to the tile space
#     _, tds = loader.get_tile_count(fsize)
#     anno_rot_ds = anno_rot / tds
#
#     # apply the annotation to the neuron density calculation
#     for i in range(nd.shape[0]):
#         for j in range(nd.shape[1]):
#             if not ray_tracing(i, j, anno_rot_ds):
#                 nd[i][j] = 0
#
#     # sig = nd[:, 350]
#     sig = nd[:, 460]
#     sig -= np.min(sig)
#
#     return nd, sig
#
# def plot_rotation(frame, anno, centroid, r, tissue_mask, tissx, tissy, frame_rot, anno_rot):
#
#     fig = plt.figure(constrained_layout=True)
#     gs = fig.add_gridspec(1, 3)
#
#     # plot the original image, with the geometry of the polygon and rotation
#     axs = []
#     ax = fig.add_subplot(gs[0, 0])
#     axs.append(ax)
#     ax.imshow(frame)
#     plot_vertices(ax, anno)
#     ax.plot(centroid[0], centroid[1], '.')
#     circle = plt.Circle(centroid, r, color='black', fill=False)
#     ax.add_patch(circle)
#
#     # plot the tissue mask, with the detected line of best fit
#     ax = fig.add_subplot(gs[0, 1])
#     axs.append(ax)
#     ax.imshow(tissue_mask)
#     ax.plot(tissx, tissy, color='red')
#
#     # plot the rotated image and annotation
#     ax = fig.add_subplot(gs[0, 2])
#     axs.append(ax)
#     ax.imshow(frame_rot)
#     plot_vertices(ax, anno_rot)
#
#     for ax in axs:
#         ax.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_aspect('equal')
#         ax.set_xlim([0, frame.shape[1]])
#         ax.set_ylim([frame.shape[0], 0])
#     plt.show()
#
# def linear_regression(x, y):
#     n = x.shape[0]
#     ss_xy = np.sum(y * x) - n * np.mean(y) * np.mean(x)
#     ss_xx = np.sum(x**2) - n * np.mean(x)**2
#     m = ss_xy/ss_xx
#     b = np.mean(y) - m * np.mean(x)
#     return m, b
#
# # TODO: use jit throughout this code!
# # NOTE: might be hard to implement this on other's computers...
# #          either will have to spend a lot of time installing everything, OR just have it run on my computer. I'm a fan of creating a web server
# @jit(nopython=True)
# def ray_tracing(x,y,poly):
#     n = len(poly)
#     inside = False
#     p2x = 0.0
#     p2y = 0.0
#     xints = 0.0
#     p1x,p1y = poly[0]
#     for i in range(n+1):
#         p2x,p2y = poly[i % n]
#         if y > min(p1y,p2y):
#             if y <= max(p1y,p2y):
#                 if x <= max(p1x,p2x):
#                     if p1y != p2y:
#                         xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
#                     if p1x == p2x or x <= xints:
#                         inside = not inside
#         p1x,p1y = p2x,p2y
#
#     return inside
#
# def plot_vertices(ax, v):
#     for i in range(v.shape[0] - 1):
#         ax.plot([v[i][0], v[i+1][0]], [v[i][1], v[i+1][1]], color='red')


if __name__ == "__main__":
    main(sys.argv)
