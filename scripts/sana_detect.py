#!/usr/local/bin/python3.9

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np

# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, get_stain_threshold, create_mask
from sana_loader import Loader
from sana_geo import Polygon

# this script loads a series of slides and ROIs within the given slides
# it loads the data within ROIs, and generates a probability map representing
# the prob of each pixel containing significant staining
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the slide files to process
    slides = []
    for list_f in args.lists:
        slides += sana_io.read_list_file(list_f)
    if len(slides) == 0:
        print("**> No Slides Found")
        parser.print_usage()
        exit()

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        print('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i+1, len(slides)), flush=True)

        # get the prob. map frames to process
        fpath = sana_io.get_fpath(os.path.dirname(slide_f),
                                  args.idir, args.rdir)
        id = sana_io.get_slide_id(os.path.basename(slide_f))
        frames = sorted([os.path.join(fpath, f) \
                         for f in os.listdir(fpath) if 'PROB' in f and id in f])

        # loop through the prob. maps
        annos = []
        for frame_i, frame_f in enumerate(frames):
            print('----> Processing Frame (%d/%d)' % \
                  (frame_i+1, len(frames)), flush=True)

            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % frame_i,
                fpath=args.idir, rpath=args.rdir)
            writer = DataWriter(data_f)

            # load the frame
            loader = Loader(slide_f)
            lvl = writer.data['lvl']
            loc = writer.data['loc']
            loader.set_lvl(lvl)
            converter = loader.converter
            frame = Frame(frame_f, lvl, converter)

            # load the mask
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % frame_i,
                fpath=args.idir, rpath=args.rdir)
            mask = Frame(mask_f, lvl, converter)

            if args.method == 'naive':

                # get the threshold
                try:
                    stain_threshold = int(args.threshold)
                except ValueError:
                    stain_threshold = get_stain_threshold(
                        frame, mi=args.stain_min, mx=args.stain_max)
                writer.data['stain_threshold'] = stain_threshold

                # threshold the image
                frame.threshold(stain_threshold, 0, 1)

                # mask the frame
                frame.mask(mask)

                # detect the objects in the thresholded frame
                frame.get_contours()

                # filter the contours
                # TODO: args should be cmdl args
                # TODO: args should be in microns
                frame.filter_contours(min_body_area=10)

                # get the detected objects
                detections = frame.get_body_contours()

                # get the polygons, translate back to the origin
                polygons = [d.polygon for d in detections]
                for i in range(len(polygons)):
                    polygons[i].translate(-writer.data['loc'])
                    polygons[i] = polygons[i].connect()

            else:
                pass

            # convert detected polygons to annotations
            for polygon in polygons:
                anno = polygon.to_annotation(slide_f,"Neuron")
                annos.append(anno)
        #
        # end of frames loop

        # save the results
        out_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.odir)
        sana_io.write_annotations(out_f, annos)
    #
    # end of slides loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True,
                        help="filelists containing .svs files")
    parser.add_argument('-idir', type=str, default="",
                        help="location to read prob. maps from")
    parser.add_argument('-odir', type=str, default="",
                        help="location to write detections to")
    parser.add_argument('-rdir', type=str, default="",
                        help="directory path to replace")
    parser.add_argument('-method', type=str, default="naive",
                        help="detection method to use")
    parser.add_argument('-threshold', type=str, default="kittler",
                        help="value of threshold, or algorithm to use")
    parser.add_argument('-stain_min', type=int, default=0,
                        help="min value for thresholding algorithms")
    parser.add_argument('-stain_max', type=int, default=255,
                        help="max value for thresholding algorithms")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
