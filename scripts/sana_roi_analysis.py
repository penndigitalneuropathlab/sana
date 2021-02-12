
import os
import sys
import argparse
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_io
import sana_geo
import sana_proc
from sana_loader import Loader

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str)
    args = parser.parse_args()

    if args.filename is None:
        filename = DEF_FILENAME
    else:
        filename = args.filename
    dname = os.path.dirname(filename).replace('images', 'annotations')
    fname = os.path.basename(filename).replace('.svs', '.svs.json')
    a = os.path.join(dname, 'Li_annotations-' + fname)

    # initialize the Loader
    loader = Loader(filename)
    loader.set_lvl(0)

    # pre-calculate the tissue threshold
    tissue_threshold = sana_proc.get_tissue_threshold(filename)

    # loop through the annotations
    annos = sana_io.read_qupath_annotations(a, loader.mpp, loader.ds)
    for anno in annos:

        # get the thumbnail resolution detections of the tissue in the ROI
        layer_0, tissue_mask = sana_proc.detect_layer_0_roi(filename, anno, tissue_threshold)

        # TODO: need to crop the image as well based on the radius?
        # get a rotated version of the roi defined by the annotation and the
        #  direction of the tissue boundary
        frame, anno, frame_rot, anno_rot, a, b = sana_proc.rotate_roi(
            loader, anno, layer_0)

        centroid, radius = anno.centroid()

        # finally, plot the results
        plot1(frame, anno, centroid, radius,
              tissue_mask, a, b, frame_rot, anno_rot)
        plt.show()

def plot1(frame, anno, centroid, radius,
          tissue_mask, a, b, frame_rot, anno_rot):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # plot the color and grayscale histos
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(frame.img)
    plot_vertices(ax0, anno.vertices())
    ax0.plot(centroid[0], centroid[1], '.')
    circle = plt.Circle(centroid, radius, color='black', fill=False)
    ax0.add_patch(circle)
    ax0.set_title('User Annotated ROI')

    # plot the tissue mask, with the detected line of best fit
    sana_geo.rescale(a, 2)
    sana_geo.rescale(b, 2)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(tissue_mask.img)
    ax1.plot((a[0], b[0]), (a[1], b[1]), color='red')
    ax1.set_title('Tissue Detection')

    # plot the rotated neuron mask and annotation
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(frame_rot.img)
    plot_vertices(ax2, anno_rot.vertices())
    ax2.set_title('Orthogonal Neuron Detection Mask')

    for ax in [ax0, ax2]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlim([0, frame.size[0]])
        ax.set_ylim([frame.size[1], 0])
    plt.show()

def plot_vertices(ax, v):
    for i in range(v.shape[0] - 1):
        ax.plot([v[i][0], v[i+1][0]], [v[i][1], v[i+1][1]], color='red')

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
