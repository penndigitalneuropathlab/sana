#!/usr/bin/env python

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

            # load the original mask
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % frame_i,
                fpath=args.idir, rpath=args.rdir)
            orig_mask = Frame(mask_f, lvl, converter)

            # generate the supplementary masks
            anno_f = sana_io.create_filepath(
                slide_f, ext='.json',
                fpath=args.adir, rpath=args.rdir)
            masks = []
            for roi_class in args.roi_classes:
                rois = sana_io.read_annotations(anno_f, roi_class)
                for roi in rois:
                    roi.translate(loc)
                masks.append(create_mask(
                    rois, frame.size(), lvl, converter, y=255))
            #
            # end of supplementary masks

            # get the threshold
            stain_threshold = None
            try:
                stain_threshold = int(args.threshold)
            except ValueError:
                pass
            if stain_threshold is None:
                try:
                    stain_threshold = get_stain_threshold(
                        frame, mi=args.stain_min, mx=args.stain_max)
                except:
                    pass
            if stain_threshold is None:
                continue
                
            writer.data['stain_threshold'] = stain_threshold

            # threshold the image
            frame.threshold(stain_threshold, 0, 255)

            # TODO: apply a morph closing to remove holes inside of objects
            pass

            # mask the frame by the original mask
            frame_orig_mask = frame.copy()
            frame_orig_mask.mask(orig_mask)

            # get the total area of the masked frame
            total_area = np.sum(orig_mask.img / 255)

            # get the area of positive data
            area = np.sum(frame_orig_mask.img / 255)

            # TODO: run this on path data, compare to ordinal vs. qupath ao
            # calculate %AO
            ao = area / total_area
            writer.data['ao'] = ao

            # apply the supplementary masks and get the %AO
            aos = []
            for mask in masks:
                frame_mask = frame.copy()
                frame_mask.mask(mask)
                total_area = np.sum(mask.img / 255)
                area = np.sum(frame_mask.img / 255)
                ao = area / total_area
                aos.append(ao)
            writer.data['aos_list'] = aos

            # save the results
            out_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_THRESH' % frame_i,
                fpath=args.idir, rpath=args.rdir)
            frame_orig_mask.save(out_f)
            writer.write_data()
        #
        # end of roi loop
    #
    # end of stain loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True,
                        help="filelists containing .svs files")
    parser.add_argument('-idir', type=str, default="",
                        help="location to write probability maps")
    parser.add_argument('-adir', type=str, default="",
                        help="location to write probability maps")
    parser.add_argument('-roi_classes', type=str, nargs='*', default=[],
                        help="ROI to use to generate the mask, if wanted")
    parser.add_argument('-rdir', type=str, default="",
                        help="directory path to replace")
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
