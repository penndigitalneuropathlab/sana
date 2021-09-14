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
from sana_frame import Frame, mean_normalize, create_mask
from sana_color_deconvolution import StainSeparator
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

        # get the annotation file
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # initalize the Loader
        loader = Loader(slide_f)
        print("--> Processing: %s (%d/%d)" % \
              (os.path.basename(slide_f), slide_i+1, len(slides)), flush=True)
        converter = loader.converter
        loader.set_lvl(args.lvl)

        # load the ROIs to process
        rois = sana_io.read_annotations(anno_f, args.roi_class)
        if rois is None or len(rois[0]) == 0:
            print("****> Skipping: No ROIs Found!")
            continue

        # loop through the ROIs
        for roi_i, roi in enumerate(rois):
            print('----> Processing ROI %d/%d' % \
                  (roi_i+1, len(rois)), flush=True)

            # get the prob map output filename
            out_f = sana_io.create_filepath(
                slide_f, ext=args.ofiletype, suffix='_%d_PROB' % roi_i,
                fpath=args.odir, rpath=args.rdir)

            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.odir, rpath=args.rdir)
            writer = DataWriter(data_f)

            # get the mask output filename
            mask_f = sana_io.create_filepath(
                slide_f, ext=args.ofiletype, suffix='_%d_MASK' % roi_i,
                fpath=args.odir, rpath=args.rdir)

            # define the bounding box of the ROI, translate to local origin
            loc, size = roi.bounding_centroid()
            roi.translate(loc)
            writer.data['loc'] = loc
            writer.data['size'] = size

            # load the frame based on the bounding box of the ROI
            frame = loader.load_frame(loc, size)

            # TODO: should have a -mask_tissue arg
            # create a binary mask to remove data outside the ROI
            mask = create_mask([roi], frame.size(), frame.lvl, frame.converter)

            # TODO: need to get this from the stain type (NeuN -> H-DAB)
            # separate the target stain from the image
            stain = 'H-DAB'
            separator = StainSeparator(stain, args.target, od=False, gray=True)
            frame_stain = Frame(separator.run(frame.img)[0],
                                loader.lvl, loader.converter)

            # inverse image to create a stain prob map
            frame_stain.img = 255 - frame_stain.img

            # TODO: when to do this in the script
            # mask out unwanted data
            frame_stain.mask(mask)

            # normalize image by the local means to remove
            #  inconsistent background staining
            frame_stain = mean_normalize(loader, frame_stain)

            # anistrophic diffusion filter to smooth the interior of staining
            # TODO: run this
            # frame_stain.anisodiff()

            # finally, save the results
            frame_stain.save(out_f)
            mask.save(mask_f)
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
    parser.add_argument('-lvl', type=int, default=2, choices=[0, 1, 2],
                        help="specify the slide level to process on")
    parser.add_argument('-target', type=str, choices=['DAB', 'HEM'],
                        help="stain to process in the image")
    parser.add_argument('-roi_class', type=str, default='ROI',
                        help="class name of the ROI to process")
    parser.add_argument('-adir', type=str, default="",
                        help="location of files containing ROIs")
    parser.add_argument('-odir', type=str, default="",
                        help="location to write probability maps")
    parser.add_argument('-rdir', type=str, default="",
                        help="directory path to replace")
    parser.add_argument('-ofiletype', type=str, default='.png',
                        help="output image file extension")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
