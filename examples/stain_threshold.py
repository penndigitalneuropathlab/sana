
import os
import sys
import argparse
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_io
import sana_proc
from sana_loader import Loader
from sana_framer import Framer
from sana_thresholder import TissueThresholder, StainThresholder
from sana_detector import TissueDetector#, NeuronDetector

SRC = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA = os.path.join(SRC, 'examples', 'data')

DEF_FILENAME = os.path.join(DATA, 'images', '2016-146-44F_R_MFC_NeuN_1K_11-04-2020_RL.svs')
DEF_ADIR = os.path.join(DATA, 'annotations')
DEF_STAIN = 'H-DAB'
DEF_TARGET = 'DAB'

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str, default=DEF_FILENAME)
    parser.add_argument('-adir', type=str, default=DEF_ADIR)
    parser.add_argument('-stain', type=str, default=DEF_STAIN)
    parser.add_argument('-target', type=str, default=DEF_TARGET)
    args = parser.parse_args()

    if args.filename is None:
        filename = DEF_FILENAME
    else:
        filename = args.filename
    slide_f = args.filename

    # get the annotation file
    # NOTE: this will either be a series of candidate ROIs, or simply
    #        the ROI to process itself
    anno_f = sana_io.slide_to_anno(args.filename, adir=args.adir)

    # initialize the Loader
    loader = Loader(slide_f)
    loader.set_lvl(1)

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

    # generate a current resolution tissue mask from the rotated ROI
    tissue_mask = sana_proc.get_tissue_mask(
        frame_rot, tissue_threshold, loader.mpp, loader.ds, loader.lvl)

    # get the stain separated roi
    frame_stain = sana_proc.separate_roi(
        frame_rot, args.stain, args.target)
    frame_plot = frame_stain.copy()

    # threshold the stain separated roi
    sana_proc.threshold_roi(frame_stain, tissue_mask, blur=1)

    # perform the tiling to get the stain density
    density_short = sana_proc.density_roi(
        loader, frame_stain, tsize=(500, 50), tstep=(20, 20))
    density_long = sana_proc.density_roi(
        loader, frame_stain, tsize=(500, 200), tstep=(20, 20))

    plot1(frame_plot, frame_stain, density_short, density_long)

    # finally, show the plot
    plt.show()

def plot1(frame_stain, frame_thresh, density_short, density_long):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 4)

    # plot the stain separated frame
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(frame_stain.img)
    ax0.axis('off')
    ax0.grid('off')

    # plot the thresholded stain separated frame
    ax1 = fig.add_subplot(gs[0,1])
    ax1.matshow(frame_thresh.img)
    ax1.axis('off')
    ax1.grid('off')
    ax1.get_shared_x_axes().join(ax0, ax1)
    ax1.get_shared_y_axes().join(ax0, ax1)

    # plot the short density analysis
    ax1 = fig.add_subplot(gs[0,2])
    ax1.imshow(density_short.img)
    ax1.axis('off')
    ax1.grid('off')

    # plot the long density analysis
    ax2 = fig.add_subplot(gs[0,3])
    ax2.imshow(density_long.img)
    ax2.axis('off')
    ax2.grid('off')
    ax2.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_y_axes().join(ax1, ax2)

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
