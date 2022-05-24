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

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

from sana_processors.NeuN_processor import NeuNProcessor
from sana_processors.SMI32_processor import SMI32Processor
from sana_processors.calretinin_processor import calretininProcessor
from sana_processors.MBP_processor import MBPProcessor
from sana_processors.SMI35_processor import SMI35Processor
from sana_processors.parvalbumin_processor import parvalbuminProcessor
from sana_processors.HDAB_processor import HDABProcessor

# instantiates a Processor object based on the antibody of the svs slide
# TODO: where to put this
def get_processor(fname, frame, debug=False, debug_fibers=False):
    antibody = sana_io.get_antibody(fname)
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
    if antibody == 'TDP43MP' or antibody == 'AT8':
        return HDABProcessor(fname, frame, debug)
#
# end of get_processor

# this script loads a series of slides and landmark vectors
# it loads and processes the data near the vector, and generates
# a various segmentation ROI's representing GM, WM and subGM regions
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

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

        # read the landmark vectors from the annotation file
        # NOTE: this is just for validation purposes
        #        will eventually delete!!
        if 'NeuN' in slide_f or 'SMI32' in slide_f or 'SMI94' in slide_f or 'parvalbumin' in slide_f and False:
            landmarks_f = sana_io.create_filepath(
                slide_f, ext='.npy', fpath=args.adir, rpath=args.rdir)
            try:
                v = np.load(landmarks_f)
            except:
                print('Skipping:', slide_f)
                continue

            # convert the array to a Line array
            v = Line(v[:,0], v[:,1], False, 0)            
        
            layers = None
            # layers_f = sana_io.create_filepath(
            #     slide_f, ext='.json', fpath='annotations/rois')
            # annos = sana_io.read_annotations(layers_f)
            # LAYERS = ['Layer I', 'Layer II', 'Layer III', 'Layer IV', 'Layer V', 'Layer VI']
            # layers = []
            # for layer in LAYERS:
            #     try:
            #         a = [a for a in annos if a.class_name == layer][0]
            #         layers.append(a)
            #     except:
            #         layers.append(None)
            # if any([x is None for x in layers]):
            #     continue
            
        # just load the landmarks that were annotated
        # TODO: clean this up
        else:    
            landmarks_f = sana_io.create_filepath(slide_f, ext='.json', fpath=args.adir)
            if not os.path.exists(landmarks_f):
                continue
            v = sana_io.read_annotations(landmarks_f, class_name=args.vector_class)
            if len(v) == 0:
                continue
            v = v[0]
            
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

        # TODO: loop through multiple vectors!

        # initialize the Params IO object, this stores params
        # relating to loading/processing the Frame
        params = Params()
        
        # load the frame into memory
        # TODO: clean this up!
        print('Loading the Frame', flush=True)
        frame, v, layers, M, orig_frame = loader.from_vector(params, v, layers)

        # get the processor object
        processor = get_processor(slide_f, frame, args.debug)

        # TODO: error checking
        if processor is None:
            continue

        v = np.copy(v)

        # run the GM segmentation algorithm!
        processor.run_segment(args.odir, params, v)
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
    parser.add_argument('-debug', action='store_true')
    return parser
#
# end of cmdl_parser

if __name__ == '__main__':
    main(sys.argv)

#
# end of file


