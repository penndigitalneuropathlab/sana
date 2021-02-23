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
DEF_DETECT_ROI = False
DEF_FILETYPE = '.png'
DEF_LEVEL = 2
DEF_TSIZE = (500, 300)
DEF_TSTEP = (25, 25)
DEF_NAME_ROI = '_ORIG'
DEF_NAME_STAIN = '_STAIN'
DEF_NAME_THRESH = '_THRESH'
DEF_STAIN = 'H-DAB'
DEF_TARGET = 'DAB'
DEF_ADIR = None
DEF_ODIR = None
DEF_RDIR = None

def main(argv):
    info = [['filename', 'tissue threshold', 'stain threshold', 'angle', 'tsize', 'tstep']]
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

        # get the annotation file
        # NOTE: this will either be a series of candidate ROIs, or simply
        #        the ROI to process itself
        anno_f = sana_io.slide_to_anno(slide_f, adir=args.adir, rdir=args.rdir)

        # initialize the Loader
        loader = Loader(slide_f)
        loader.set_lvl(args.level)

        # pre-calculate the tissue threshold
        print("----> Pre-Calculating the Tissue/Slide Threshold")
        tissue_threshold = sana_proc.get_tissue_threshold(slide_f)

        # TODO: need to handle writing multiple ROI's per image
        # loop through the annotations
        annos = sana_io.read_qupath_annotations(anno_f, loader.mpp, loader.ds)
        for anno_i, anno in enumerate(annos):
            print("----> Processing ROI (%d/%d)" % (anno_i+1, len(annos)))

            # get the thumbnail resolution detections of the tissue in the ROI
            print("------> Detecting Layer 0 Boundary in ROI")
            layer_0, tissue_mask_tb = sana_proc.detect_layer_0_roi(
                slide_f, anno, tissue_threshold)

            if layer_0.n == 0:
                print("WARNING: No Slide Background Detected in ROI. Skipping ROI...")
                continue

            # TODO: need to crop the image as well based on the radius?
            # get a rotated version of the roi defined by the annotation and the
            #  direction of the tissue boundary
            print("------> Rotating/Cropping ROI by Angle of Layer 0 Boundary")
            frame, anno, frame_rot, anno_rot, a, b, angle, trans_loc0, trans_loc1 = \
                sana_proc.rotate_roi(loader, anno, layer_0)
            centroid, radius = anno.centroid()

            # generate a current resolution tissue mask from the rotated ROI
            print("------> Generating Tissue Mask for the ROI")
            tissue_mask = sana_proc.get_tissue_mask(
                frame_rot, tissue_threshold, loader.mpp, loader.ds, loader.lvl)

            # decide whether or not to process this frame
            # TODO: not implemented yet, need to check the parallelness
            if args.detect_roi:
                pass

            if args.write_roi:
                print("--------> Writing Rotated/Cropped ROI")
                roi_f = sana_io.get_ofname(slide_f, args.filetype,
                                           DEF_NAME_ROI,
                                           args.odir, args.rdir)
                frame_rot.save(roi_f)

            # perform color deconvolution on the ROI
            print("------> Separating %s from the %s Stained ROI" % \
                  (args.target, args.stain))
            frame_stain = sana_proc.separate_roi(
                frame_rot, args.stain, args.target)
            if args.write_stain:
                print("--------> Writing Stain Separated ROI")
                stain_f = sana_io.get_ofname(slide_f, args.filetype,
                                             DEF_NAME_STAIN,
                                             args.odir, args.rdir)
                frame_stain.save(stain_f)

            # perform the thresholding on the stain separated ROI
            print("------> Thresholding the Stain Separated ROI")
            stain_threshold = sana_proc.threshold_roi(
                frame_stain, tissue_mask, blur=1)
            if args.write_thresh:
                print("--------> Writing Thresholded ROI")
                thresh_f = sana_io.get_ofname(slide_f, args.filetype,
                                              DEF_NAME_THRESH,
                                              args.odir, args.rdir)
                frame_stain.save(thresh_f)

            # perform the tiled density calculation
            frame_density, tds = sana_proc.density_roi(
                loader, frame_stain, args.tsize, args.tstep, round=True)
            print("------> Calculating Density of Thresholded ROI")
            if args.write_density:
                print("--------> Writing Density")
                density_f = sana_io.get_ofname(slide_f, ftype=args.filetype,
                                               odir=args.odir, rdir=args.rdir)
                frame_density.save(density_f)

            # store information about the processing that occurred
            info.append(map(str, [slide_f, loader.mpp, loader.ds, loader.lvl, list(trans_loc0), list(trans_loc1), list(centroid), radius, angle, list(tds), tissue_threshold, stain_threshold, angle, list(args.tsize), list(args.tstep)]))
        #
        # end of annos loop
    #
    # end of slides loop

    # write the info related to the processing parameters
    info_of = sana_io.get_ofname(
        'information.csv', odir=args.odir, rdir=args.rdir)
    fp = open(info_of, 'w')
    for x in info:
        fp.write('\t'.join(x) + '\n')
    fp.close()

#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser(usage=open(USAGE).read())
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument('-detect_roi', type=bool, default=DEF_DETECT_ROI,
                        help="filter candidate ROIs into a single ROI")
    parser.add_argument('-level', type=int, default=DEF_LEVEL,
                        help="specify the slide level to process on\n[default: 0]")
    parser.add_argument('-tsize', type=int, nargs=2, default=DEF_TSIZE,
                        help="tile size for analysis\n[default: 500 300]")
    parser.add_argument('-tstep', type=int, nargs=2, default=DEF_TSTEP,
                        help="tile step for analysis\n[default: 25 25]")
    parser.add_argument('-stain', type=str, default=DEF_STAIN,
                        help="define the Stain type (H-DAB, HED)\n[default: H-DAB]")
    parser.add_argument('-target', type=str, default=DEF_TARGET,
                        help="define the Stain target (HEM, DAB, EOS, RES, ALL) \n[default: DAB]")
    parser.add_argument('-adir', type=str, default=DEF_ADIR,
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default=DEF_ODIR,
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default=DEF_RDIR,
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-filetype', type=str, default=DEF_FILETYPE,
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
