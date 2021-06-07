#!/usr/local/bin/python3.9

import os
import sys
import cv2
import time
from copy import copy
import numpy as np
import argparse
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import pyplot as plt

import sana_io
from sana_frame import Frame
from sana_geo import Array, Point, Polygon, ray_tracing
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader
from sana_thresholds import knn

USAGE = os.path.expanduser(
    '~/neuropath/src/sana/scripts/usage/sana_gm_segmentation.usage')
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

        # get the annotation file
        anno_f = sana_io.convert_fname(
            slide_f, ext='.json', odir=args.adir, rdir=args.rdir)

        # get the output annotation file
        out_f = sana_io.convert_fname(
            slide_f, ext='.json', odir=args.odir, rdir=args.rdir)

        # define the target stain to process based on the slide filename
        if 'NeuN' in slide_f:
            target = 'DAB'
        else:
            target = 'HEM'

        # initialize the slide loader, set the level
        try:
            loader = Loader(slide_f)
        except:
            print("Slide not found:", slide_f)
            continue
        print("--> Processing: %s (%d/%d)" % \
              (os.path.basename(slide_f), slide_i+1, len(slides)), flush=True)
        converter = loader.converter
        loader.set_lvl(args.lvl)

        # load the crude ROIs to process
        rois = sana_io.read_annotations(anno_f, args.roi_class)
        if rois is None or len(rois[0]) == 0:
            print("Skipping Annotations: %s" % anno_f)
            continue

        # loop through the crude ROIs
        segs = []
        for roi_i, roi in enumerate(rois[0]):
            print('----> Processing ROI %d/%d' % \
                  (roi_i, len(rois[0])), flush=True, end="")

            # get the roi metrics file
            metrics_f = sana_io.convert_fname(
                slide_f, ext='.csv', suffix='_%d_DAT' % roi_i,
                odir=args.odir, rdir=args.rdir)

            # get the tissue mask filename
            tissue_f = sana_io.convert_fname(
                out_f, ext='.npy', suffix='_%d_TISSUE' % roi_i)

            # get the tiled density measurements
            frame_density, frame_size, roi, angle, loc, crop_loc, ds = \
                get_tile_features(args, loader, out_f, metrics_f, roi_i, roi)

            # get the tissue mask
            frame_tissue = Frame(
                np.load(tissue_f), loader.lvl, loader.converter)

            # apply KNN to get the 3 most prominent values
            # NOTE: in DAB this will be Slide/WM, low GM, and high GM
            # NOTE: in HEM this will be Slide, GM, and WM
            seg, thresholds = knn(frame_density, 3)
            d_seg = seg
            seg = Frame(seg)
            if target == 'DAB':
                seg.threshold(thresholds[0], x=0, y=1)
                seg = seg.img[:, :, 0]
            else:
                seg.threshold(thresholds[1], x=1, y=0)
                seg = seg.img[:, :, 0]

            # define the min thickness for layer 0 and layer 6 in microns
            N0 = Array([10.], is_micron=True)
            converter.to_pixels(N0, loader.lvl)
            N6 = Array([500.], is_micron=True)
            converter.to_pixels(N6, loader.lvl)
            N6 = N6 / ds[1]
            N0, N6 = int(N0[0]), int(N6[0])
            l0, l6 = [], []

            # loop from bottom to top in each column, find the first instance
            # of positive thresholded data
            for i in range(seg.shape[1]):
                x = seg[:, i]
                found = False
                for j in range(x.shape[0]-1, -1, -1):
                    if all(x[j-N6:j] == 1):
                        l6.append(j)
                        found = True
                        break
                if not found:
                    l6.append(x.shape[0]-1)
            l6 = np.array(l6)

            # remove the WM using the detected boundary
            for i in range(frame_density.shape[1]):
                frame_density[l6[i]:, i] = 0

            # if we're using HEM stain, need to re-run knn
            # to get slide, low GM, and high GM
            if target == 'HEM':
                seg, thresholds = knn(frame_density, 3)
                seg = Frame(seg)
                seg.threshold(thresholds[0], x=0, y=1)
                seg = seg.img[:, :, 0]

            # loop from bottom to top of each column, find the first
            # instance of zeroed slide
            for i in range(0, frame_tissue.img.shape[1], 20):
                x = frame_tissue.img[:, i, 0]
                found = False
                for j in range(x.shape[0]//2, -1, -1):
                    if all(x[j-N0:j] == 0):
                        l0.append(j / ds[1])
                        found = True
                        break
                if not found:
                    l0.append(0)
            l0 = np.array(l0)
            d_seg = seg

            l0 = signal.resample(l0, l6.shape[0])

            # TODO: interpolate then median filter!
            n = l0.shape[0]
            x = np.arange(0, n-1, 0.1)
            f0 = interp1d(np.arange(l0.shape[0]), l0)
            f6 = interp1d(np.arange(l6.shape[0]), l6)
            l0 = f0(x)
            l6 = f6(x)

            l0 = uniform_filter1d(l0, x.shape[0]//8)
            l6 = uniform_filter1d(l6, x.shape[0]//4)

            # TODO: filter by the original ROI

            # build the gm segmentation by combining layers 0 and 6
            x = np.concatenate([x[::-1], x])
            y = np.concatenate([l0[::-1], l6])
            seg = Polygon(x, y, False, loader.lvl).astype(float)

            # revert from the tile dimension
            seg[:, 0] *= ds[0]
            seg[:, 1] *= ds[1]

            # translate to origin before cropping
            seg.translate(-crop_loc)

            # rotate back to original orientation
            roi.translate(loc)
            seg = seg.rotate(roi.centroid()[0], -angle)

            # translate to slide origin
            seg.translate(-loc)

            # convert to max resolution
            loader.converter.rescale(seg, 0)

            # connect the last point to the first point
            seg = seg.connect()

            # finally, store the detected segmentation
            segs.append(seg)

            print('')
        #
        # end of crude ROI loop

        # write the detected segmentations to output annotation file
        sana_io.write_annotations(out_f, segs, class_name=args.seg_class)
    #
    # end of slides loop
#
# end of main

def get_tile_features(args, loader, out_f, metrics_f, roi_i, roi):

    # define file location for the density file and roi metrics
    density_f = sana_io.convert_fname(
        out_f, ext='.npy', suffix='_%d_DENSITY' % roi_i)
    size_f = sana_io.convert_fname(
        out_f, ext='.npy', suffix='_%d_SIZE' % roi_i)

    # check whether we need to generate the tiled density
    if os.path.exists(density_f) and not any([args.proc_stain, args.proc_filt, args.proc_tiles]):
        frame_density = np.load(density_f)
        frame_size = np.load(size_f)
        angle, loc, crop_loc, ds, _, _ = sana_io.read_metrics_file(metrics_f)
    else:

        # get the filtered and processed frame
        frame_filt, roi, angle, loc, crop_loc = get_filtered_frame(
            args, loader, out_f, metrics_f, roi_i, roi)

        # frame_stain, roi, angle, loc, crop_loc = get_stain_separated_frame(
        #     args, loader, out_f, metrics_f, roi_i, roi)

        # load the tiles in the frame
        tsize = Point(args.tsize[0], args.tsize[1], True)
        tstep = Point(args.tstep[0], args.tstep[1], True)
        tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        tiler.set_frame(frame_filt, pad=True)
        ds = tiler.ds
        tiles = tiler.load_tiles()

        # orig_tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        # orig_tiler.set_frame(frame_stain, pad=True)
        # orig_tiles = orig_tiler.load_tiles()

        # write the tile downsampling to the metrics file
        sana_io.write_metrics_file(metrics_f, ds=ds)

        # prepare the tile feature matrices
        frame_density = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
        frame_size = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)

        if 'NeuN' in out_f:
            radius = 9
        else:
            radius = 5
        gap = int(radius//3)

        # loop through the tiles
        for j in range(tiles.shape[1]):
            for i in range(tiles.shape[0]):

                # create a frame for the tile
                s = ' -- %d/%d' % (j*tiles.shape[0]+i,
                    tiles.shape[0]*tiles.shape[1])
                print(s + "\b"*len(s), flush=True, end="")

                # detect the centers of each of the cells
                frame = Frame(tiles[i][j], loader.lvl, loader.converter)
                centers = frame.detect_cell_centers(radius, gap)

                # calculate the size of the cells based on the centers
                frame = Frame(tiles[i][j], loader.lvl, loader.converter)
                sizes = frame.detect_cell_sizes(centers)

                # filter out very small objects in the NeuN stain
                if 'NeuN' in out_f:
                    d = [(centers[i], sizes[i]) for i in range(len(centers)) if sizes[i] > 5]
                    if len(d) == 0:
                        centers, sizes = [], []
                    else:
                        centers, sizes = zip(*d)

                # calculate the density as num cells per mm^2
                count = len(centers)
                density = 1000 * count / (tsize[0] * tsize[1])

                # calculate the average cell size
                fsizes = [s for s in sizes if s != 0]
                if len(fsizes) < 1:
                    size = 0
                else:
                    size = np.std(sizes)
                    # size = np.mean(fsizes)

                # store the density and size features
                frame_density[i][j] = density
                frame_size[i][j] = size

                # if (j > 6 and i > 9) or True:
                #     ax = plt.gca()
                #     ax.imshow(255-orig_tiles[i][j], cmap='gray')
                #     circles = [plt.Circle(centers[i], sizes[i], color='red', fill=False) for i in range(len(centers)) if sizes[i] != 0]
                #     for c in circles:
                #         ax.add_patch(c)
                #     plt.show()

        # finally, save the tile metrics
        np.save(density_f, frame_density)
        np.save(size_f, frame_size)

        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(frame_density)
        # axs[1].imshow(frame_size)
        # plt.show()

    return frame_density, frame_size, roi, angle, loc, crop_loc, ds

def get_filtered_frame(args, loader, out_f, metrics_f, roi_i, roi):

    # define the file location for the processed frame
    filt_f = sana_io.convert_fname(
        out_f, ext='.npy', suffix='_%d_FILT' % roi_i)

    # check whether we need to generate the frame
    if os.path.exists(filt_f) and not any([args.proc_filt, args.proc_stain]):
        frame_filt = Frame(np.load(filt_f), loader.lvl, loader.converter)
        angle, loc, crop_loc, _, _, _ = sana_io.read_metrics_file(metrics_f)
    else:

        # get the stain separated frame
        frame_stain, roi, angle, loc, crop_loc = get_stain_separated_frame(
            args, loader, out_f, metrics_f, roi_i, roi)

        # inverse image to create a cell prob. heatmap
        frame_stain.img = 255 - frame_stain.img

        # anistrophic diffusion filter to smooth cell interiors
        frame_filt = frame_stain.copy()
        frame_filt.anisodiff()
        frame_filt.round()

        # threshold for the cells
        thresh = frame_filt.copy().get_stain_threshold(mi=8)
        frame_filt.threshold(thresh, x=0, y=255)
        sana_io.write_metrics_file(metrics_f, stain_threshold=thresh)

        # morphological filtering to clean up the background
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        frame_filt.img = cv2.erode(frame_filt.img, kern)[:, :, None]

        # finally, save the frame
        np.save(filt_f, frame_filt.img)

    return frame_filt, roi, angle, loc, crop_loc

def get_stain_separated_frame(args, loader, out_f, metrics_f, roi_i, roi):

    # define file location for the stain separated frame and roi metrics
    stain_f = sana_io.convert_fname(
        out_f, ext='.npy', suffix='_%d_STAIN' % roi_i)
    tissue_f = sana_io.convert_fname(
        out_f, ext='.npy', suffix='_%d_TISSUE' % roi_i)

    # check whether we need to generate the stain separated frame
    if os.path.exists(stain_f) and not any([args.proc_stain]):
        frame_stain = Frame(np.load(stain_f), loader.lvl, loader.converter)
        angle, loc, crop_loc, _, _, _ = sana_io.read_metrics_file(metrics_f)
    else:
        # pre-calculate the tissue threshold
        thumbnail = loader.thumbnail.copy()
        tissue_threshold = thumbnail.get_tissue_threshold(blur=5)

        # define the bounding box of the ROI, translate to local origin
        c, r = roi.centroid()
        loc = Point(c[0]-r, c[1]-r, False, 0)
        size = Point(2*r, 2*r, False, 0)
        roi.translate(loc)
        sana_io.write_metrics_file(metrics_f, loc=loc)

        # calculate the angular rotation of the tissue/slide boundary
        frame_thumb = loader.load_frame(loc, size, lvl=loader.lc-1)
        loader.converter.rescale(roi, loader.lc-1)
        angle = frame_thumb.tissue_angle(tissue_threshold, roi, 1e5, 1e5)
        loader.converter.rescale(roi, loader.lvl)
        loader.converter.rescale(loc, loader.lvl)
        loader.converter.rescale(size, loader.lvl)
        if angle is None:
            angle = 0
        sana_io.write_metrics_file(metrics_f, angle=angle)

        # load the frame based on the radius of the ROI
        frame = loader.load_frame(loc, size)

        # rotate the frame around the centroid of the ROI
        rot_center = roi.centroid()[0]
        roi = roi.rotate(rot_center, angle)
        frame = frame.rotate(angle)

        # crop the rotated frame to remove unneeded data
        crop_loc, crop_size = roi.bounding_box()
        frame = frame.crop(crop_loc, crop_size)
        roi.translate(crop_loc)
        sana_io.write_metrics_file(metrics_f, crop_loc=crop_loc)

        # generate the tissue mask from the rotated ROI
        tissue_mask = frame.copy()
        tissue_mask.detect_tissue(tissue_threshold, 1e5, 1e5)
        tissue_mask.to_mask()
        sana_io.write_metrics_file(metrics_f, tissue_threshold=tissue_threshold)

        # separate the stain from the ROI using color deconvolution
        if 'NeuN' in out_f:
            target = 'DAB'
        else:
            target = 'HEM'
        separator = StainSeparator(args.stain, target, ret_od=False)
        frame_stain = separator.run(frame.img)[0]
        frame_stain = Frame(frame_stain, loader.lvl, loader.converter)
        if frame_stain.img.dtype == np.uint8:
            frame_stain.to_gray()

        # finally, save the frame
        np.save(tissue_f, tissue_mask.img)
        np.save(stain_f, frame_stain.img)
        roi.translate(-crop_loc)
        roi = roi.rotate(rot_center, -angle)
        roi.translate(-loc)
    return frame_stain, roi, angle, loc, crop_loc

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
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-ofiletype', type=str, default='.png',
                        help="output image file extension\n[default: .png]")
    parser.add_argument('-proc_stain', action='store_true',
                        help='program re-calculates the original frame')
    parser.add_argument('-proc_filt', action='store_true',
                        help='program re-calculates the processed frame')
    parser.add_argument('-proc_tiles', action='store_true',
                        help="program re-calculates tile features")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
