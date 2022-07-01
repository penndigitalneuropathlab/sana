#!/usr/bin/env python3

# git bash command to run sana_process
# debug_sana_process -adir ./data/meguro_pilot/annotations/ -odir ./data/meguro_pilot/outputs_v1 -lists ./data/meguro_pilot/lists/all.list -roi_type GM -main_class GM -debug_level full

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
from sana_geo import transform_poly
from sana_frame import Frame
from sana_processors.NeuN_processor import NeuNProcessor
from sana_processors.SMI32_processor import SMI32Processor
from sana_processors.calretinin_processor import calretininProcessor
from sana_processors.MBP_processor import MBPProcessor
from sana_processors.SMI35_processor import SMI35Processor
from sana_processors.parvalbumin_processor import parvalbuminProcessor
from sana_processors.meguro_processor import meguroProcessor
from sana_processors.AT8_processor import AT8Processor

# debugging modules
from sana_geo import plot_poly
from matplotlib import pyplot as plt

# instantiates a Processor object based on the antibody of the svs slide
# TODO: where to put this
def get_processor(fname, frame, debug=False, debug_fibers=False):
    try:
        antibody = sana_io.get_antibody(fname)
    except:
        antibody = ''
    if antibody == 'NeuN':
        return NeuNProcessor(fname, frame, debug)
    if antibody == 'SMI32':
        return SMI32Processor(fname, frame, debug)
    if antibody == 'CALR6BC':
        return calretininProcessor(fname, frame, debug)
    if antibody == 'parvalbumin':
        return parvalbuminProcessor(fname, frame, debug)
    if antibody == 'SMI94':
        return MBPProcessor(fname, frame, debug, debug_fibers)
    if antibody == 'SMI35':
        return SMI35Processor(fname, frame, debug)
    if antibody == 'MEGURO':
        return meguroProcessor(fname, frame, debug)
    if antibody == 'AT8':
        return AT8Processor(fname, frame, debug)
#
# end of get_processor

# this script loads a series of slides and ROIs within the given slides
# it loads and processes the data within ROIs, and calculates the percentage
# of positive pixels in the ROIs
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # Dictionary of logging levels
    level_config = {'full': logging.DEBUG,         # value: 10
                    'normal': logging.INFO,        # value: 20
                    # 'WARNING': logging.WARNING,  # value: 30
                    'quiet': logging.ERROR,        # value: 40
                    # 'CRITICAL': logging.CRITICAL # value: 50
                    }

    # Configure logger object

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(funcName)s :: %(message)s')

    file_handler = logging.FileHandler('log.log',mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Set logging level from commandline
    level = level_config.get(args.debug_level.lower())

    logger.setLevel(level)

    # get all the slide files to process
    slides = sana_io.get_slides_from_lists(args.lists)
    if len(slides) == 0:
        logger.error("No Slides Found")
        parser.print_usage()
        exit()
    logger.debug('Number of slides found: %d' % len(slides))

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        t0 = time.time()
        # progress messaging
        logger.info('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i+1, len(slides)))

        # get the annotation file containing ROIs
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # make sure the file exists, else we skip this slide
        if not os.path.exists(anno_f):
            logger.warning('Annotation file %s doesn\'t exist\n' % anno_f)
            continue

        # initialize the Loader object for loading Frames
        try:
            loader = Loader(slide_f)
        except Exception as e:
            print(e)
            logger.warning('Could\'t load .svs file: %s' % e)
            continue

        # set the image resolution level
        loader.set_lvl(args.lvl)

        # load the main roi(s) from the json file
        main_rois = sana_io.read_annotations(anno_f, class_name=args.main_class)

        # load the sub roi(s) from the json file
        sub_rois = []
        for sub_class in args.sub_classes:
            sub_rois += sana_io.read_annotations(anno_f, sub_class)

        # logger.debug for # of main_roi and sub_roi
        logger.debug('Number of main_rois found: %d' % len(main_rois))
        logger.debug('Number of sub_rois found: %d' % len(sub_rois))

        # loop through main roi(s)
        for main_roi_i, main_roi in enumerate(main_rois):

            # progress messaging
            logger.info('----> Processing Frame (%d/%d)' % \
                  (main_roi_i+1, len(main_rois)))

            # initialize the Params IO object, this will store parameters
            # relating to the loading/processing of the Frame, as well as
            # the various AO results
            params = Params()

            # create the output directory path
            # NOTE: XXXX-XXX-XXX/antibody/region/ROI_0/
            bid = sana_io.get_bid(slide_f)
            antibody = sana_io.get_antibody(slide_f)
            region = sana_io.get_region(slide_f)
            roi_id = '%s_%d' % ('ROI', main_roi_i)
            odir = sana_io.create_odir(args.odir, bid)
            odir = sana_io.create_odir(odir, antibody)
            odir = sana_io.create_odir(odir, region)
            odir = sana_io.create_odir(odir, roi_id)

            logger.debug('Output directory successfully created: %s' % odir)

            padding = 400

            # load the frame into memory using the main roi
            if args.roi_type == 'GM':

                # rotate/translate the coord. system to retrieve the frame from
                # the slide. the frame will be orthogonalized such that CSF is
                # at the top and WM is at the bottom of the image.
                frame = loader.load_gm_frame(params, main_roi, padding=padding, debug=level)
            else:
                # just translates the coord. system, no rotating or cropping
                frame = loader.load_roi_frame(params, main_roi, padding=padding, debug=level)

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
                slide_f, frame)

            if processor is None:
                continue

            # run the processes for the antibody
            processor.run(odir, params, main_roi, sub_rois)
            logger.info('Runtime: %0.2f (sec)' % (time.time()-t0))

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
        '-sub_classes', type=str, nargs='*', default=[],
        help="class names of ROIs inside the main ROI to separately process")
    parser.add_argument(
        '-debug_level', type=str, default='normal',
        help="Logging debug level", choices=['full', 'normal', 'quiet'])

    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
