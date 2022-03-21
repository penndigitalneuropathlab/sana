#!/usr/bin/env python

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, get_stain_threshold, create_mask
from sana_loader import Loader
<<<<<<< HEAD
from matplotlib import pyplot as plt
=======
from sana_geo import transform_poly
>>>>>>> af7da75ed9e64e4ccda6e042a67f287eed3e2c47

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

        anno_f = sana_io.create_filepath(slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)
        
        if not os.path.exists(anno_f):
            continue
        
        rois = sana_io.read_annotations(anno_f, class_name=args.roi_class)
        for roi_i, roi in enumerate(rois):
            print('----> Processing Frame (%d/%d)' % \
                  (roi_i+1, len(rois)), flush=True)            
            
            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            writer = DataWriter(data_f)

            # load the frame
            frame_f = sana_io.create_filepath(slide_f, ext='.png', fpath=args.idir, rpath=args.rdir,
                                              suffix='_%d_PROB' % roi_i)
            if not os.path.exists(frame_f):
                continue
            loader = Loader(slide_f)
            lvl = writer.data['lvl']
            loc = writer.data['loc']
            loader.set_lvl(lvl)
            converter = loader.converter
            frame = Frame(frame_f, lvl, converter)

            roi = transform_poly(
                roi, writer.data['loc'], writer.data['crop_loc'],
                writer.data['M1'], writer.data['M2'])
            
            # load the original mask
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            orig_mask = Frame(mask_f, lvl, converter)

            # generate the supplementary masks
            anno_f = sana_io.create_filepath(
                slide_f, ext='.json',
                fpath=args.adir, rpath=args.rdir)
            masks = []
            for roi_class in args.roi_classes:
                rois = sana_io.read_annotations(anno_f, roi_class)
                for roi_i in range(len(rois)):
                    rois[roi_i] = transform_poly(
                        rois[roi_i], writer.data['loc'], writer.data['crop_loc'],
                        writer.data['M1'], writer.data['M2'])

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
            writer.data['area'] = total_area

            # apply the supplementary masks and get the %AO
            aos, areas = [], []
            for mask in masks:
                frame_mask = frame.copy()
                frame_mask.mask(mask)
                total_area = np.sum(mask.img / 255)
                area = np.sum(frame_mask.img / 255)
                ao = area / total_area
                areas.append(total_area)
                aos.append(ao)
            writer.data['aos_list'] = aos
            writer.data['areas_list'] = areas

            # save the results
            out_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_THRESH' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            qc_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_QC' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            orig_frame_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_ORIG' % roi_i,
                fpath=args.idir, rpath=args.rdir)            
            orig_frame = Frame(orig_frame_f, loader.lvl, loader.converter)
            img1 = orig_frame.img.astype(float)/255
            img2 = frame.img.astype(float)/255
                
            z = np.zeros((frame.img.shape[0], frame.img.shape[1], 1)).astype(float)
            img2 = np.concatenate((img2, z, z), axis=-1)
            
            alpha = 0.7
            a2 = np.clip(img2 - (1-alpha), 0, 1)
                
            img3 = (1.0-a2) * img1 + a2 * img2
            img3 = np.rint(img3*255).astype(np.uint8)
            
            Image.fromarray(img3).save(qc_f)
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
                        help="location to read prob. maps from")
    parser.add_argument('-adir', type=str, default="",
                        help="location to read masks from")
    parser.add_argument('-rdir', type=str, default="",
                        help="directory path to replace")
    parser.add_argument('-roi_class', type=str, default='ROI')
    parser.add_argument('-roi_classes', type=str, nargs='*', default=[],
                        help="ROI to use to generate the mask, if wanted")
    parser.add_argument('-threshold', type=str, default="kittler",
                        help="value of threshold, or algorithm to use")
    parser.add_argument('-stain_min', type=int, default=1,
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
