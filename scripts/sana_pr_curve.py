#!/usr/bin/env python

# system packages
import os
import sys
import argparse

# installed packages
import numpy as np
from matplotlib import pyplot as plt

# custom packages
from sana_io import get_anno_files, read_annotations, read_list_file, create_filepath
import sana_io
from sana_loader import Loader
from sana_geo import plot_poly

# cmdl interface help messages
SLIST_HELP = "List of slides to be processed"
LVL_HELP = "Specify which pixel resolution to use while processing"
REFDIR_HELP = "Directory containing reference annotation files"
HYPDIR_HELP = "Directory containing hypothesis annotation files"
REFCLASS_HELP = "List of reference class names to use"
HYPCLASS_HELP = "List of hypothesis class names to use"
ROICLASS_HELP = "List of roi class names to load, only annotations in these rois will be considered"
IOUTHRESH_HELP = "List of IOU Thresholds to use while scoring"

# defines the location of the usage text file for cmdl interface
USAGE = os.path.join(os.environ.get('SANAPATH'),
                     'scripts', 'usage', 'sana_pr_curve.usage')

# calculates the IOU score of 2 Polygons
# NOTE: this uses Shapely, converts to shapely objects for easy calculations
def iou(x, y):
    x = x.copy().to_shapely()
    y = y.copy().to_shapely()
    i = x.intersection(y).area
    u = x.union(y).area
    return i/u
#
# end of iou

def score(ref_annos, hyp_annos, iou_threshold, args):

    # dict for storing the ref annotations that have been used for scoring
    seen = {}

    # list for storing the TP/FP decision for each hyp anno
    results = []

    # loop through the hyp annotations
    for hyp in hyp_annos:

        # initalize seen dict for new filenames
        if hyp.file_name not in seen:
            seen[hyp.file_name] = []

        # calculate the IOU scores between all refs and the given hyp
        scores = []
        for ref in ref_annos:

            # only get IOU score if ref is in the same file as hyp
            if os.path.basename(ref.file_name) == os.path.basename(hyp.file_name):
                try:
                    scores.append(iou(ref, hyp))
                except:
                    scores.append(0)

        # no ref in the current hyp's filename, FP
        if len(scores) == 0:
            results.append(False)

        else:
            # get the maximum IOU score
            ind = np.argmax(scores)

            # hyp is overlapping with a ref
            if scores[ind] >= iou_threshold:

                # ref already seen, FP
                if ind in seen[hyp.file_name]:
                    results.append(False)


                # ref not seen, TP
                else:
                    results.append(True)
                    seen[hyp.file_name].append(ind)

                #
                # end of seen checking

            # hyp not overlapping with a ref, FP
            else:
                plot(args, hyp.file_name)
                results.append(False)
            #
            # end of iou_threshold checking
        #
        # end of available ref checking
    #
    # end of hyp annotations loop

    return results
#
# end of score

# returns all annos within any of the givenrois
# NOTE: if no rois given, returns all annos
def check_inside(rois, annos):

    # if no rois given, return all annos
    if len(rois) == 0:
        return annos
    else:

        # loop through the annos
        inds = []
        for i in range(len(annos)):

            # loop through the rois
            flag = False
            for roi in rois:

                # check if the anno is fully inside the roi
                if annos[i].filter(roi).shape == annos[i].shape:
                    flag = True
                    break
            #
            # end of roi loop

            if flag:
                inds.append(i)
        #
        # end of annos loop

        return [annos[i] for i in inds]
    #
    # end of roi length checking
#
# end of check_inside

def get_annos(args):

    # get all annotation files in the ref and hyp dirs
    ref_files = sorted(get_anno_files(args.refdir))

    hyp_files = sorted(get_anno_files(args.hypdir))

    # loop through annotation files
    ref_annos, hyp_annos = [], []
    for rf in ref_files:
        for hf in hyp_files:

            # make sure there is a one to one match between ref and hyp dirs
            if os.path.basename(rf) == os.path.basename(hf):

                # load the ref rois, if given
                if args.roiclass is None:
                    rois = []
                else:
                    rois = []
                    for roi_class in args.roiclass:
                        rois += read_annotations(rf, class_name=roi_class)
                #
                # end of roi reading

                # load the ref annotations with any of the given ref classes
                if args.refclass is None:
                    ra = read_annotations(rf)
                else:
                    ra = []
                    for rclass in args.refclass:
                        ra += read_annotations(rf, class_name=rclass)
                #
                # end of ref anno reading

                # load the hyp annotations and confidences with a given hyp class
                if args.hypclass is None:
                    ha = read_annotations(hf)
                else:
                    ha = []
                    for hclass in args.hypclass:
                        ha += read_annotations(hf, class_name=hclass)
                #
                # end of hclass loop

                # keep only the annotations within a given roi
                ref_annos += check_inside(rois, ra)
                hyp_annos += check_inside(rois, ha)

                break
            #
            # end of file matching check
        #
        # end of hyp file loop
    #
    # end of ref file loop

    # sort the hyp annotations from most to least confident
    hyp_annos.sort(key=lambda x: x.confidence, reverse=True)

    return ref_annos, hyp_annos
#
# end of get_annos

def pr(ref_annos, hyp_annos, iou_threshold, args):

    # get the TP/FP decisions for all the hyp annos
    results = score(ref_annos, hyp_annos, iou_threshold, args)

    # calculate rolling sum of TPs from most to least confident
    cumulative_tp = np.cumsum(results)

    # calculate rolling precision from most to least confident
    # prec = TP / Number of Detections
    precision = cumulative_tp / np.arange(1, len(results)+1)

    # calculate rolling recall from most to least confident
    # rec = TP / Number of Positives
    recall = cumulative_tp / len(ref_annos)

    # calculate area under the curve
    ap = np.sum(precision[1:] * (recall[1:] - recall[:-1]))

    return precision, recall, ap

#COME BACK TO HERE TO CONTINUE WORKING ON FUNCTION
def plot(args, slide):
    print(slide)
    # find the annotation file based on the slide filename
    image = sana_io.create_filepath(slide, ext='.svs', fpath=args.refdir)
    print(image)
    exit()
    # # access slides
    # slides = sana_io.read_list_file(args.slist)
    #
    # # loop through each of the slides in the list of slides
    # for slide in slides:
    # initialize loader, this will allow us to load slide data into memory
    loader = Loader(image)
    converter = loader.converter

    # tell the loader which pixel resolution to use
    loader.set_lvl(args.lvl)

    # find the annotation file based on the slide filename
    anno_fname = sana_io.create_filepath(slide, ext='.json', fpath=args.refdir)

    print(anno_fname)
    exit()
    # load the annotations from the annotation file
    ROIS = sana_io.read_annotations(anno_fname, class_name='ROI')

    # rescale the ROI annotations to the given pixel resolution
    for ROI in ROIS:
        if ROI.lvl != args.lvl:
            converter.rescale(ROI, args.lvl)

    # load the annotations from the annotation file
    REF_ANNOS = sana_io.read_annotations(anno_fname, class_name='ARTIFACT')

    # rescale the ROI annotations to the given pixel resolution
    for REF_ANNO in REF_ANNOS:
        if REF_ANNO.lvl != args.lvl:
            converter.rescale(REF_ANNO, args.lvl)

    #find the hypothesis annotation file based on the slide filename
    hypothesis_fname = sana_io.create_filepath(slide, ext='.json', fpath=args.hypdir)


    # load the annotations from the annotation file
    HYP_ANNOS = sana_io.read_annotations(hypothesis_fname, class_name='ARTIFACT')

    # rescale the ROI annotations to the given pixel resolution
    for HYP_ANNO in HYP_ANNOS:
        if HYP_ANNO.lvl != args.lvl:
            converter.rescale(HYP_ANNO, args.lvl)

    for ROI in ROIS:
        # get the top left coordinate and the size of the bounding centroid
        loc, size = ROI.bounding_box()

        frame = loader.load_frame(loc, size)

        fig, axs = plt.subplots(1,2, sharex=False, sharey=False)

        # Plotting the processed frame and processed frame with clusters
        axs[0].imshow(frame.img, cmap='gray')
        for REF_ANNO in REF_ANNOS:
            REF_ANNO.translate(loc)
            plot_poly(axs[0], REF_ANNO, color='red')

        axs[1].imshow(frame.img, cmap='gray')
        for HYP_ANNO in HYP_ANNOS:
            HYP_ANNO.translate(loc)
            plot_poly(axs[1], HYP_ANNO, color='blue')
        plt.show()

#
# end of pr
# this script takes a group of ref and hyp annotations, and outputs 1 or more
# PR curves based on the given cmdl arguments
def main(argv):

    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-slist', type=str, default='/Volumes/RESEARCH/BCHE300/all.list',help=SLIST_HELP)
    parser.add_argument('-lvl', type=int, default=0,help=LVL_HELP)
    parser.add_argument('-refdir', type=str, required=True, help=REFDIR_HELP)
    parser.add_argument('-hypdir', type=str, required=True, help=HYPDIR_HELP)
    parser.add_argument('-refclass', type=str, nargs='*', help=REFCLASS_HELP)
    parser.add_argument('-hypclass', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-roiclass', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-iouthresh', type=float, nargs='*', default=[0.5],
                        help=IOUTHRESH_HELP)
    parser.add_argument('-odir', type=str, default='.')
    parser.add_argument('-title', type=str, default="")
    args = parser.parse_args()

    fig, ax = plt.subplots(1,1)

    # load the valid ref/hyp annos and the hyp confidences
    ref_annos, hyp_annos = get_annos(args)

    # loop through all provided iou thresholds
    for iou_threshold in args.iouthresh:

        # generate precision/recall curve, calculate the average precision
        precision, recall, ap = pr(ref_annos, hyp_annos, iou_threshold, args)

        # plot the curve
        ax.plot(recall, precision,
                label='IOU = %d%% -- AP = %.2f%%' % (100*iou_threshold, 100*ap))
    #
    # end of iou_threshold loop

    # finally, show the plot
    ax.set_title(args.title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(args.odir, 'pr_curves.png'), dpi=300)
#
# end of main

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
