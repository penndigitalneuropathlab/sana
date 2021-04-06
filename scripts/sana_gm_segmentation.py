#!/usr/local/bin/python3.9

import os
import sys
import time
from copy import copy
import numpy as np
import argparse
from scipy import signal
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_io
from sana_frame import Frame
from sana_geo import Point, Polygon, ray_tracing
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader

USAGE = os.path.expanduser(
    '~/neuropath/src/sana/scripts/usage/sana_roi_analysis.usage')
DEF_NAME_ROI = '_ORIG'
DEF_NAME_STAIN = '_STAIN'
DEF_NAME_THRESH = '_THRESH'

def main(argv):
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
        loader = Loader(slide_f)
        converter = loader.converter
        loader.set_lvl(args.lvl)

        # get the annotation file
        anno_f = sana_io.slide_to_anno(slide_f, adir=args.adir, rdir=args.rdir)

        # TODO: this threshold is not consistent!!
        # pre-calculate the tissue threshold
        print("----> Pre-Calculating the Tissue/Slide Threshold")
        thumbnail = loader.thumbnail.copy()
        tissue_threshold = thumbnail.get_tissue_threshold(blur=5)

        # loop through the GM ROIs and generate the segmentations
        gm_segs = []
        gm_seg_metrics = []
        gm_rois, _, gm_names = sana_io.read_annotations(
            anno_f, args.roi_class)
        for gm_roi_i, gm_roi in enumerate(gm_rois):
            print("\n----> Segmenting GM ROI (%d/%d)" % \
                  (gm_roi_i+1, len(gm_rois)))
            t0 = time.time()
            t1 = time.time()

            # define the bounding box of the ROI
            c, r = gm_roi.centroid()
            roi_loc = Point(c[0]-r, c[1]-r, False, 0)
            roi_size = Point(2*r, 2*r, False, 0)

            # load the thumbnail resolution roi, get the direction of the tissue
            print("------> Detecting Tissue Rotation: ", end="")
            gm_roi.translate(roi_loc)
            roi_thumb = loader.load_frame(roi_loc, roi_size, lvl=loader.lc-1)
            angle = roi_thumb.tissue_angle(tissue_threshold, gm_roi, 1e5, 1e5)
            converter.rescale(gm_roi, loader.lvl)
            converter.rescale(roi_loc, loader.lvl)
            converter.rescale(roi_size, loader.lvl)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            gm_roi_rot = gm_roi.rotate(gm_roi.centroid()[0], angle)
            print("------> Loading Frame: ", end="", flush=True)
            frame = loader.load_frame(roi_loc, roi_size)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            print("------> Rotating Frame: ", end="", flush=True)
            frame_rot = frame.rotate(angle)
            crop_loc, crop_size = gm_roi_rot.bounding_box()
            frame_rot = frame_rot.crop(crop_loc, crop_size)
            gm_roi_rot.translate(crop_loc)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            # generate the tissue mask from the rotated ROI
            print("------> Detecting Tissue in ROI: ", end="", flush=True)
            tissue_mask = frame_rot.copy()
            tissue_mask.detect_tissue(tissue_threshold, 1e5, 1e5)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            print("------> Masking out Background: ", end="", flush=True)
            tissue_mask.to_mask(x=0, y=255)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            # separate the stain from the ROI using color deconvolution
            print("------> Stain Separating: ", end="", flush=True)
            separator = StainSeparator(args.stain, args.target, ret_od=False)
            frame_stain = separator.run(frame_rot.img)[0]
            frame_stain = Frame(frame_stain, loader.lvl, converter)
            frame_stain.to_gray()
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            print("------> Stain Thresholding: ", end="", flush=True)
            stain_threshold = frame_stain.copy().get_stain_threshold(tissue_mask)
            frame_thresh = frame_stain.copy()
            frame_thresh.threshold(stain_threshold, x=1, y=0)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            print("------> Calculating Stain Density: ", end="", flush=True)

            # load the tiles
            tsize = Point(args.tsize[0], args.tsize[1], True)
            tstep = Point(args.tstep[0], args.tstep[1], True)
            tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
            tiles = tiler.load_tiles(frame_thresh, pad=True)

            # calculate the density of the stain in the frame
            frame_density = Frame(
                np.mean(tiles, axis=(2,3)), loader.lvl, loader.converter)
            print("%.3f sec" % (time.time() - t1))
            t1 = time.time()

            # detect the layer 6 boundary from the density frame
            # NOTE: this is scaled up from the tile dimension
            layer_6_threshold = 0.2
            x_6, y_6 = frame_density.copy().detect_layer_6(layer_6_threshold)
            x_6 = np.rint(x_6 * tiler.ds[0]).astype(np.int)
            y_6 = np.rint(y_6 * tiler.ds[1]).astype(np.int)

            # detect the layer 0 boundary from the tissue mask
            x_0, y_0 = tissue_mask.copy().detect_layer_0(xvals=x_6)
            x_0 = np.rint(x_0).astype(np.int)
            y_0 = np.rint(y_0).astype(np.int)

            print(len(x_6), len(x_0))
            # smooth the detections
            y_0 = signal.medfilt(y_0, 31)
            y_6 = signal.medfilt(y_6, 31)

            # filter the detections by the given roi, also filter so that
            #   layer 0 or layer 6 don't extend past each other
            v_0 = np.array((x_0, y_0)).T
            v_6 = np.array((x_6, y_6)).T
            v_0 = [v for v in v_0 if ray_tracing(v[0], v[1], gm_roi_rot)]
            v_6 = [v for v in v_6 if ray_tracing(v[0], v[1], gm_roi_rot)]
            left, right = max(v_0[0][0], v_6[0][0]), min(v_0[-1][0], v_6[-1][0])
            v_0 = np.array([v for v in v_0 if v[0] >= left and v[0] <= right])
            v_6 = np.array([v for v in v_6 if v[0] >= left and v[0] <= right])

            # build the gm segmentation by combining layers 0 and 6
            x = np.concatenate([v_0[::-1][:, 0], v_6[:, 0]])
            y = np.concatenate([v_0[::-1][:, 1], v_6[:, 1]])
            gm_seg = Polygon(x, y, False, loader.lvl).astype(float)

            # revert to original orientation, location, and resolution
            gm_seg = gm_seg.rotate(gm_roi_rot.centroid()[0], -angle)
            gm_seg.translate(-(crop_loc + roi_loc))
            loader.converter.rescale(gm_seg, 0)

            # connect the last point to the first point
            gm_seg = gm_seg.connect()
            gm_segs.append(gm_seg)

            # gm_plot = copy(gm_seg)
            # loader.converter.rescale(gm_plot, loader.lvl)
            # gm_plot.translate(roi_loc)
            # fig, axs = plt.subplots(2,2)
            # axs = axs.ravel()
            # axs[0].imshow(frame_stain.img)
            # for i in range(gm_roi_rot.shape[0]-1):
            #     axs[0].plot(gm_roi_rot[i:i+2, 0], gm_roi_rot[i:i+2, 1], color='black')
            # axs[1].imshow(frame_thresh.img)
            # axs[2].imshow(frame_density.img)
            # axs[3].imshow(frame.img)
            # for i in range(gm_plot.shape[0]-1):
            #     axs[3].plot(gm_plot[i:i+2, 0], gm_plot[i:i+2, 1], color='black')
            # plt.show()

            # gm_seg, angle, metrics = sana_proc.segment_gm(
            #     args, slide_f, gm_roi, tissue_threshold, count=gm_roi_i)
            # gm_seg_metrics.append(metrics)
        #
        # end of gm_rois loop

        sana_io.append_annotations(anno_f, gm_segs,
                                   class_name=args.seg_class,
                                   anno_names=gm_names)
    #
    # end of slides loop

        # store information about the processing that occurred
        # info.append(map(str, [slide_f, loader.mpp, loader.ds, loader.lvl, list(trans_loc0), list(trans_loc1), list(centroid), radius, angle, list(tds), tissue_threshold, stain_threshold, angle, list(args.tsize), list(args.tstep)]))
        #
        # end of annos loop
    #
    # end of slides loop

    # # write the info related to the processing parameters
    # info_of = sana_io.get_ofname(
    #     'information.csv', odir=args.odir, rdir=args.rdir)
    # fp = open(info_of, 'w')
    # for x in info:
    #     fp.write('\t'.join(x) + '\n')
    # fp.close()

#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser(usage=open(USAGE).read())
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument('-lvl', type=int, default=2,
                        help="specify the slide level to process on\n[default: 0]")
    parser.add_argument('-seg_class', type=str, default='GM_SEG',
                        help="class name of the GM Segmentation")
    parser.add_argument('-roi_class', type=str, default='GM_ROI',
                        help="class name of the GM Crude ROI")
    parser.add_argument('-tsize', type=int, nargs=2, default=(500, 300),
                        help="tile size for analysis\n[default: 500 300]")
    parser.add_argument('-tstep', type=int, nargs=2, default=(25, 25),
                        help="tile step for analysis\n[default: 25 25]")
    parser.add_argument('-stain', type=str, default="H-DAB",
                        help="define the Stain type (H-DAB, HED)\n[default: H-DAB]")
    parser.add_argument('-target', type=str, default="DAB",
                        help="define the Stain target (HEM, DAB, EOS, RES, ALL) \n[default: DAB]")
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-ofiletype', type=str, default='.png',
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
