#!/usr/bin/env python3

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np

# custom modules
import sana_io
from sana_params import Params
from sana_loader import Loader
from sana_geo import transform_poly
from sana_antibody_processor import get_processor

# debugging modules
from sana_geo import plot_poly
from matplotlib import pyplot as plt

# this script loads a series of slides and ROIs within the given slides
# it loads and processes the data within ROIs, and calculates the percentage
# of positive pixels in the ROIs
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the slide files to process
    slides = sana_io.get_slides_from_lists(args.lists)
    if len(slides) == 0:
        print("***> ERROR: No Slides Found")
        parser.print_usage()
        exit()

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):

        # progress messaging
        print('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i+1, len(slides)), flush=True)

        # get the annotation file containing ROIs
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # make sure the file exists, else we skip this slide
        if not os.path.exists(anno_f):
            print('***> Warning: Annotation file doesn\'t exist, skipping...')
            continue

        # initialize the Loader object for loading Frames
        try:
            loader = Loader(slide_f)
        except:
            print('***> Warning: Could\'t load .svs file, skipping...')
            continue

        # set the image resolution level
        loader.set_lvl(args.lvl)
        
        # load the main roi(s) from the json file
        main_rois = sana_io.read_annotations(anno_f, class_name=args.main_class)

        # load the sub roi(s) from the json file
        sub_rois = []
        for sub_class in args.sub_classes:
            sub_rois += sana_io.read_annotations(anno_f, sub_class)
        
        # loop through main roi(s)
        for main_roi_i, main_roi in enumerate(main_rois):

            # progress messaging
            print('----> Processing Frame (%d/%d)' % \
                  (main_roi_i+1, len(main_rois)), flush=True)
            
            # initialize the Params IO object, this will store parameters
            # relating to the loading/processing of the Frame, as well as
            # the various AO results
            params = Params()

            # create the output directory path
            # TODO: do we want aid instead? would make less dirs, but then need
            #       to put the 03F part somewhere i think...
            cid = sana_io.get_cid(slide_f)
            antibody = sana_io.get_antibody(slide_f)
            odir = sana_io.create_odir(args.odir, cid)
            odir = sana_io.create_odir(odir, '%s_%d' % (antibody, main_roi_i))
            
            # load the frame into memory using the main roi
            if args.roi_type == 'GM':

                # rotate/translate the coord. system to retrieve the frame from
                # the slide. the frame will be orthogonalized such that CSF is
                # at the top and WM is at the bottom of the image.
                frame = loader.load_gm_frame(params, main_roi)
            else:

                # just translates the coord. system, no rotating or cropping
                frame = loader.load_roi_frame(params, main_roi)

            # transform the main ROI to the Frame's coord. system
            main_roi = transform_poly(
                main_roi,
                params.data['loc'], params.data['crop_loc'],
                params.data['M1'], params.data['M2']
            )

            # transform the sub ROIs to the Frame's coord. system
            for sub_roi_i in range(len(sub_rois)):
                sub_rois[sub_roi_i] = transform_poly(
                    sub_rois[sub_roi_i],
                    params.data['loc'], params.data['crop_loc'],
                    params.data['M1'], params.data['M2']
                )
                
            # get the processor object
            processor = get_processor(slide_f, frame)

            # run the processes for the antibody
            processor.run(odir, params, main_roi, sub_rois)
        #
        # end of main_rois loop
    #
    # end of slides loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-lists', type=str, nargs='*', required=True,
        help="filelists containing .svs files")
    parser.add_argument(
        '-adir', type=str, default="",
        help="directory path containing .json files")
    parser.add_argument(
        '-odir', type=str, default="",
        help="directory path to write the results to")
    parser.add_argument(
        '-rdir', type=str, default="",
        help="directory path to replace")
    parser.add_argument(
        '-lvl', type=int, default=0,
        help="resolution level to use during processing")
    parser.add_argument(
        '-roi_type', type=str, required=True, choices=['GM', 'ROI'],
        help="type of ROI the annotations should be treated as, GM ROIs will be rotated"
    )
    parser.add_argument(
        '-main_class', type=str, default='ROI',
        help="ROI class used to load and process the Frame")
    parser.add_argument(
        '-sub_classes', type=str, nargs='*', default=[],
        help="class names of ROIs inside the main ROI to separately process")

    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
