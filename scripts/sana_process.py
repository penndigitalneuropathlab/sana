#!/usr/bin/env python3

# system modules
import os
import sys
import argparse
import logging
import time

# installed modules
import numpy as np

# custom modules
import sana_io
from sana_params import Params
from sana_loader import Loader
from sana_geo import transform_poly, transform_inv_poly
from sana_frame import Frame
from sana_logger import SANALogger

# TODO: rename processors to classifiers
from sana_processors.NeuN_processor import NeuNProcessor
from sana_processors.SMI32_processor import SMI32Processor
from sana_processors.calretinin_processor import calretininProcessor
from sana_processors.MBP_processor import MBPProcessor
from sana_processors.SMI35_processor import SMI35Processor
from sana_processors.parvalbumin_processor import parvalbuminProcessor
from sana_processors.meguro_processor import MeguroProcessor
from sana_processors.AT8_processor import AT8Processor
from sana_processors.IBA1_processor import IBA1Processor
from sana_processors.R13_processor import R13Processor

# debugging modules
from sana_geo import plot_poly
from matplotlib import pyplot as plt

# instantiates a Processor object based on the antibody of the svs slide
# TODO: where to put this
def get_processor(fname, frame, debug_level, debug_fibers=False):
    try:
        antibody = sana_io.get_antibody(fname)
    except:
        antibody = ''
    if antibody == 'NeuN':
        return NeuNProcessor(fname, frame, debug_level)
    if antibody == 'SMI32':
        return SMI32Processor(fname, frame, debug_level)
    if antibody == 'CALR6BC':
        return calretininProcessor(fname, frame, debug_level)
    if antibody == 'parvalbumin':
        return parvalbuminProcessor(fname, frame, debug_level)
    if antibody == 'SMI94':
        return MBPProcessor(fname, frame, debug_level, debug_fibers)
    if antibody == 'SMI35':
        return SMI35Processor(fname, frame, debug_level)
    if antibody == 'MEGURO':
        return MeguroProcessor(fname, frame, debug_level)
    if antibody == 'AT8':
        return AT8Processor(fname, frame, debug_level)
    if antibody == 'IBA1':
        return IBA1Processor(fname, frame, debug_level)
    if antibody == 'R13':
        return R13Processor(fname, frame, debug_level)
    return None
#
# end of get_processor

# this script loads a series of slides and ROIs within the given slides
# it loads and processes the data within ROIs, and calculates the percentage
# of positive pixels in the ROIs
def main(argv):
    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    log = SANALogger.get_sana_logger(args.debug_level)

    # get all the slide files to process
    slides = sana_io.get_slides_from_lists(args.lists)
    if len(slides) == 0:
        log.warning("No Slides Found")
        parser.print_usage()
        exit()
    log.warning('Number of slides found: %d' % len(slides))

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        # progress messaging
        log.warning('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i+1, len(slides)))

        # # get the annotation file containing ROIs
        # anno_f = sana_io.create_filepath(
        #     slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # Temporary fix for reading annotations (microglia)
        anno_f = sana_io.create_filepath(
            slide_f.replace('-21_EX', '-2021_EX'), ext='.json', fpath=args.adir, rpath=args.rdir)

        # make sure the file exists, else we skip this slide
        if not os.path.exists(anno_f):
            log.info('Annotation file %s does not exist\n' % anno_f)
            continue

        # initialize the Loader object for loading Frames
        try:
            loader = Loader(slide_f)
        except Exception as e:
            print(e)
            log.info('Could not load .svs file: %s' % e)
            continue

        # set the image resolution level
        loader.set_lvl(args.lvl)

        # load the main roi(s) from the json file
        main_rois = sana_io.read_annotations(anno_f, class_name=args.main_class, name=args.main_name)
        log.info('Number of main_rois found: %d' % len(main_rois))

        # Use this to analyze a certain frame from a slide
        # main_rois = main_rois[8:]


        # load the sub roi(s) from the json file
        # sub_rois = []
        # for sub_class in args.sub_classes:
        #     sub_rois += sana_io.read_annotations(anno_f, sub_class)

        # loop through main roi(s)
        for main_roi_i, main_roi in enumerate(main_rois):
            t0 = time.time()
            # progress messaging
            log.warning('----> Processing Frame (%d/%d)' % \
                  (main_roi_i+1, len(main_rois)))

            # load the sub roi(s) that are inside this main roi
            sub_rois = []
            for sub_class in args.sub_classes:
                rois = sana_io.read_annotations(anno_f, sub_class)
                for roi in rois:
                    if roi.inside(main_roi):
                        sub_rois.append(roi)
                        break
                # NOTE: None is used for not found sub rois since we still need to measure something
                else:
                    sub_rois.append(None)
            log.info('Number of sub_rois found: %d' % len(sub_rois))

            # load the sub roi(s) that are inside this main roi
            # NOTE: None is used for not found sub rois since we still need to measure something
            sub_rois = []
            for sub_class in args.sub_classes:
                rois = sana_io.read_annotations(anno_f, sub_class)
                for roi in rois:
                    if roi.inside(main_roi):
                        sub_rois.append(roi)
                        break
                else:
                    sub_rois.append(None)

            # initialize the Params IO object, this will store parameters
            # relating to the loading/processing of the Frame, as well as
            # the various AO results
            params = Params()

            # create odir for detection jsons
            roi_odir = sana_io.create_odir(args.odir, 'detections')
            # create the output directory path
            # NOTE: XXXX-XXX-XXX/antibody/region/ROI_0/
            try:
                bid = sana_io.get_bid(slide_f)
                antibody = sana_io.get_antibody(slide_f)
                region = sana_io.get_region(slide_f)

                if not main_roi.name:
                    roi_name = str(main_roi_i)
                else:
                    roi_name = main_roi.name

                roi_id = '%s_%s' % (main_roi.class_name, roi_name)
                odir = sana_io.create_odir(args.odir, bid)
                odir = sana_io.create_odir(odir, antibody)
                odir = sana_io.create_odir(odir, region)
                odir = sana_io.create_odir(odir, roi_id)
            except:
                odir = sana_io.create_odir(
                    args.odir, os.path.splitext(os.path.basename(slide_f))[0]+'_%d' % main_roi_i)
            log.info('Output directory successfully created: %s' % odir)

            padding = args.padding

            # load the frame into memory using the main roi
            if args.roi_type == 'GM':

                # rotate/translate the coord. system to retrieve the frame from
                # the slide. the frame will be orthogonalized such that CSF is
                # at the top and WM is at the bottom of the image.
                frame = loader.load_gm_frame(params, main_roi, padding=padding, debug_level=args.debug_level)
            else:
                # just translates the coord. system, no rotating or cropping
                frame = loader.load_roi_frame(params, main_roi, padding=padding, debug_level=args.debug_level)

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
            processor = get_processor(
                slide_f, frame, debug_level=args.debug_level)

            if processor is None:
                log.info('No processor found')
                continue
            if (main_roi_i == 0):
                first_run = True
            else:
                first_run = False

            # run the processes for the antibody
            processor.run(odir, roi_odir, first_run, params, main_roi, sub_rois)
            log.info('Runtime: %0.2f (sec)' % (time.time()-t0))

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
        help="type of ROI, GM will be rotated, ROI is only translated")
    parser.add_argument(
        '-main_class', type=str, default=None,
        help="ROI class used to load and process the Frame")
    parser.add_argument(
        '-main_name', type=str, default=None,
        help="name of annotation to match")
    parser.add_argument(
        '-sub_classes', type=str, nargs='*', default=[],
        help="class names of ROIs inside the main ROI to separately process")
    parser.add_argument(
        '-debug_level', type=str, default='normal',
        help="Logging debug level", choices=['full', 'debug', 'normal', 'quiet'])
    parser.add_argument(
        '-padding', type=int, default=0, help="Frame padding")

    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
