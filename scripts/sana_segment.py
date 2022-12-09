#!/usr/bin/env python3

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np
import cv2
from scipy.ndimage import convolve1d

# custom modules
import sana_io
from sana_params import Params
from sana_loader import Loader
from sana_tiler import Tiler
from sana_geo import Line
from sana_logger import SANALogger

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

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
from sana_processors.HDAB_processor import HDABProcessor
from sana_processors.HE_processor import HEProcessor

# instantiates a Processor object based on the antibody of the svs slide
# TODO: where to put this
def get_processor(fname, frame, logger, **kwargs):
    try:
        antibody = sana_io.get_antibody(fname)
    except:
        antibody = ''
    antibody_map = {
        'NeuN': NeuNProcessor,
        'SMI32': SMI32Processor,
        'CALR6BC': calretininProcessor,
        'parvalbumin': parvalbuminProcessor,
        'SMI94': MBPProcessor,
        'SMI35': SMI35Processor,
        'MEGURO': MeguroProcessor,
        'AT8': AT8Processor,
        'IBA1': IBA1Processor,
        'R13': R13Processor,
        'MJFR13': R13Processor,
        'HE': HEProcessor,
        '': HDABProcessor,
    }
    cls = antibody_map[antibody]
    
    return cls(fname, frame, logger, **kwargs)
#
# end of get_processor

# this script loads a series of slides and landmark vectors
# it loads and processes the data near the vector, and generates
# a various segmentation ROI's representing GM, WM and subGM regions
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    logger = SANALogger.get_sana_logger(args.debug_level)
    
    # get all the slide files to process
    slides = sana_io.get_slides_from_lists(args.lists)
    if len(slides) == 0:
        print("***> No Slides Found")
        parser.print_usage()
        exit()

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        
        # progress messaging
        print('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i, len(slides)), flush=True)

        landmarks_f = sana_io.create_filepath(slide_f, ext='.json', fpath=args.adir)
        if not os.path.exists(landmarks_f):
            continue
        vectors = sana_io.read_annotations(landmarks_f, class_name=args.vector_class)
        for vector_i, v in enumerate(vectors):

            # convert the array to a Line array
            v = Line(v[:,0], v[:,1], False, 0)
            layers = None

            # initialize the loader
            try:
                loader = Loader(slide_f)
            except Exception as e:
                print(e)
                print('***> Warning: Could\'t load .svs file, skipping...')
                continue

            # set the image resolution level
            loader.set_lvl(args.lvl)
            
            # initialize the Params IO object, this stores params
            # relating to loading/processing the Frame
            params = Params()
        
            # load the frame into memory
            # TODO: clean this up!
            print('Loading the Frame', flush=True)
            frame, v, layers, M, orig_frame = loader.from_vector(
                params, v, layers, padding=args.padding)

            # get the processor object
            kwargs = {
                'qupath_threshold': None,
                'roi_type': 'VECTOR',
            }
            processor = get_processor(slide_f, frame, logger, **kwargs)

            odir = os.path.join(args.odir, os.path.basename(slide_f).replace('.svs', '_'+str(vector_i)))
            if not os.path.exists(odir):
                os.makedirs(odir)
                
            # run the GM segmentation algorithm!
            processor.run_segment(odir, params, v, padding=args.padding)
    #
    # end of slides loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True)
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-lvl', type=int, default=0,
                        help='pixel resolution to use, 0 is maximum.')
    parser.add_argument('-vector_class', type=str, default='LANDMARK_VECTOR',
                        help='class name of the landmark vector to process')
    parser.add_argument('-padding', type=int, default=0,
                        help="amount of padding to add to the frame")
    parser.add_argument(
        '-debug_level', type=str, default='normal',
        help="Logging debug level", choices=['full', 'debug', 'normal', 'quiet'])
    return parser
#
# end of cmdl_parser

if __name__ == '__main__':
    main(sys.argv)

#
# end of file
