#!/usr/local/bin/python3.9

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

USAGE = os.path.expanduser(
    '~/neuropath/src/sana/scripts/usage/sana_roi_analysis.usage')
DEF_NAME_ROI = '_ORIG'
DEF_NAME_STAIN = '_STAIN'
DEF_NAME_THRESH = '_THRESH'

def main(argv):
    # info = [['filename', 'tissue threshold', 'stain threshold', 'angle', 'tsize', 'tstep']]
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the slide files to process
    slides = []
    for list_f in args.files:
        slides += sana_io.read_list_file(list_f)

    if len(slides) == 0:
        parser.print_usage()
        exit()

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        print("--> Processing: %s (%d/%d)" % (os.path.basename(slide_f), slide_i+1, len(slides)))
        loader = Loader(slide_f)

        # get the annotation file
        # NOTE: this will either be a series of candidate ROIs, or simply
        #        the ROI to process itself
        anno_f = sana_io.slide_to_anno(slide_f, adir=args.adir, rdir=args.rdir)

        # TODO: this threshold is not consistent!!
        # pre-calculate the tissue threshold
        print("----> Pre-Calculating the Tissue/Slide Threshold")
        tissue_threshold = sana_proc.get_tissue_threshold(slide_f)

        # case 1: GM Analysis
        if args.type == 'gm':

            # read or generate the GM segmentations
            gm_segs, _, gm_names = sana_io.read_annotations(
                anno_f, loader.mpp, loader.ds, class_name=args.gm_class)
            if args.segment_gm or len(gm_segs) == 0:

                # loop through the GM ROIs and generate the segmentations
                gm_segs = []
                gm_seg_metrics = []
                gm_rois, _, gm_names = sana_io.read_annotations(
                    anno_f, loader.mpp, loader.ds, class_name='GM_ROI')
                for gm_roi_i, gm_roi in enumerate(gm_rois):
                    print("----> Segmenting GM ROI (%d/%d)" % \
                          (gm_roi_i+1, len(gm_rois)))
                    gm_seg, angle, metrics = sana_proc.segment_gm(
                        args, slide_f, gm_roi, tissue_threshold, count=gm_roi_i)
                    gm_segs.append(gm_seg)
                    gm_seg_metrics.append(metrics)
            sana_io.append_annotations(anno_f, gm_segs,
                                       class_name=args.gm_class,
                                       anno_names=gm_names)
            # process the GM segmentations
            for gm_seg_i, gm_seg in enumerate(gm_segs):
                print("----> Processing GM Segmentation (%d/%d)" % \
                    (gm_seg_i+1, len(gm_segs)))
                measurements = sana_proc.process_gm(gm_seg)

        # case 2: WM Analysis
        if args.type == 'wm':
            pass

        # case 3: Special
        pass





            # store information about the processing that occurred
            # info.append(map(str, [slide_f, loader.mpp, loader.ds, loader.lvl, list(trans_loc0), list(trans_loc1), list(centroid), radius, angle, list(tds), tissue_threshold, stain_threshold, angle, list(args.tsize), list(args.tstep)]))
        #
        # end of annos loop
    #
    # end of slides loop

    # # write the info related to the processing parameters
    # info_of = sana_io.get_ofname(
    #     'information.csv', odir=args.odir, rdir=args.rdir)
    # fp = open(info_of, 'w')
    # for x in info:
    #     fp.write('\t'.join(x) + '\n')
    # fp.close()

#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser(usage=open(USAGE).read())
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument('-level', type=int, default=2,
                        help="specify the slide level to process on\n[default: 0]")
    parser.add_argument('-segment_gm', action='store_true',
                        help="uses the GM_ROI annotations to create segmentations")
    parser.add_argument('-type', type=str, choices=['gm', 'wm'], default='GM',
                        help="type of roi analysis to perform")
    parser.add_argument('-gm_class', type=str, default='GM_SEG',
                        help="class name of the GM region to process")
    parser.add_argument('-tsize', type=int, nargs=2, default=(500, 300),
                        help="tile size for analysis\n[default: 500 300]")
    parser.add_argument('-tstep', type=int, nargs=2, default=(25, 25),
                        help="tile step for analysis\n[default: 25 25]")
    parser.add_argument('-stain', type=str, default="H-DAB",
                        help="define the Stain type (H-DAB, HED)\n[default: H-DAB]")
    parser.add_argument('-target', type=str, default="DAB",
                        help="define the Stain target (HEM, DAB, EOS, RES, ALL) \n[default: DAB]")
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-ofiletype', type=str, default='.png',
                        help="output image file extension\n[default: .png]")
    parser.add_argument('-write_roi', action='store_true',
                        help="outputs the raw slide ROI")
    parser.add_argument('-write_stain', action='store_true',
                        help="outputs the stain separated ROI")
    parser.add_argument('-write_thresh', action='store_true',
                        help="outputs the thresholded ROI")
    parser.add_argument('-write_density', action='store_true',
                        help="outputs the tiled thresholded ROI density")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
