
import os
import sys
import argparse
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_io
import sana_geo
import sana_proc
from sana_loader import Loader

SRC = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA = os.path.join(SRC, 'examples', 'data')

DEF_FILENAME = os.path.join(DATA, 'images', '2016-146-44F_R_MFC_NeuN_1K_11-04-2020_RL.svs')
DEF_ADIR = os.path.join(DATA, 'annotations')

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str, default=DEF_FILENAME)
    parser.add_argument('-adir', type=str, default=DEF_ADIR)
    args = parser.parse_args()

    # initialize the Loader
    loader = Loader(args.filename)
    loader.set_lvl(1)

    # get the annotation file
    # NOTE: this will either be a series of candidate ROIs, or simply
    #        the ROI to process itself
    anno_f = sana_io.slide_to_anno(args.filename, adir=args.adir)

    # pre-calculate the tissue threshold
    tissue_threshold = sana_proc.get_tissue_threshold(args.filename)

    # get the ROI
    anno = sana_io.read_qupath_annotations(anno_f, loader.mpp, loader.ds)[0]

    # get the thumbnail resolution detections of the tissue in the ROI
    layer_0, tissue_mask_tb = sana_proc.detect_layer_0_roi(
                    args.filename, anno, tissue_threshold)

    # get a rotated version of the roi defined by the annotation and the
    #  direction of the tissue boundary
    frame, anno, frame_rot, anno_rot, a, b = sana_proc.rotate_roi(
        loader, anno, layer_0)
    centroid, radius = anno.centroid()

    plot1(frame, anno, centroid, radius,
          tissue_mask_tb, a, b, frame_rot, anno_rot)

def plot1(frame, anno, centroid, radius,
          tissue_mask_tb, a, b, frame_rot, anno_rot):

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
    ax0.set_xlim([0, frame.size[0]])
    ax0.set_ylim([frame.size[1], 0])

    # plot the tissue mask, with the detected line of best fit
    # TODO: y-intercept seems off?
    sana_geo.rescale(a, 2)
    sana_geo.rescale(b, 2)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.matshow(tissue_mask_tb.img)
    ax1.plot((a[0], b[0]), (a[1]-20, b[1]-20), color='red', label='Line of Best Fit')
    ax1.set_title('Tissue Detection')
    ax1.set_xlim([0, tissue_mask_tb.size[0]])
    ax1.set_ylim([tissue_mask_tb.size[1], 0])
    ax1.legend()

    # plot the rotated neuron mask and annotation
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(frame_rot.img)
    plot_vertices(ax2, anno_rot.vertices())
    ax2.set_title('Orthogonal Neuron Detection Mask')
    ax2.set_xlim([0, frame_rot.size[0]])
    ax2.set_ylim([frame_rot.size[1], 0])

    for ax in [ax0, ax1, ax2]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    plt.show()

def plot_vertices(ax, v):
    for i in range(v.shape[0] - 1):
        ax.plot([v[i][0], v[i+1][0]], [v[i][1], v[i+1][1]], color='red')

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
