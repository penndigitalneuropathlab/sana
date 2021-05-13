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
from PIL import Image

import sana_io
from sana_frame import Frame
from sana_geo import Point, Polygon, ray_tracing
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader
from sana_thresholds import knn

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

        # get the annotation file
        anno_f = sana_io.slide_to_anno(slide_f, adir=args.adir, rdir=args.rdir)

        # define the output file location
        out_f = sana_io.get_ofname(anno_f, odir=args.odir)

        if 'NeuN' in slide_f:
            target = 'DAB'
        else:
            target = 'HEM'

        print("--> Processing: %s (%d/%d)" % (os.path.basename(slide_f), slide_i+1, len(slides)))
        try:
            loader = Loader(slide_f)
        except:
            print("Slide not found:", slide_f)
            continue
        converter = loader.converter
        loader.set_lvl(args.lvl)

        # load the ROIs to process
        rois = sana_io.read_annotations(anno_f, args.roi_class)
        if rois is None or len(rois[0]) == 0:
            print(os.path.exists(anno_f))
            print("Skipping Annotations: %s" % anno_f)
            continue

        segs = []
        for roi_i, roi in enumerate(rois[0]):
            print(slide_i, '/', len(slides), '-->', roi_i, '/', len(rois[0]), flush=True)

            frame_density, metrics = get_density(args, loader, out_f, roi_i, roi)
            if metrics is None:
                continue
            angle, loc, crop_loc, ds = metrics

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

            # define the min thickness of layer 0 and layer 6
            # TODO: needs to be defined in microns, won't work with different
            #        step values
            N0, N6 = 3, 5
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
            l6 = np.array(signal.medfilt(l6, 7))
            l6 = np.rint(l6).astype(int)

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

            # loop from top to bottom of each column, find the first
            # instance of positive thresholded data
            for i in range(seg.shape[1]):
                x = seg[:, i]
                found = False
                for j in range(0, x.shape[0], 1):
                    if all(x[j:j+N0] == 1):
                        l0.append(j)
                        found = True
                        break
                if not found:
                    l0.append(0)
            l0 = np.array(signal.medfilt(l0, 7))
            l0 = np.rint(l0).astype(int)
            d_seg = seg

            # build the gm segmentation by combining layers 0 and 6
            i = np.arange(len(l0))
            x = np.concatenate([i[::-1], i])
            y = np.concatenate([l0[::-1], l6])
            seg = Polygon(x, y, False, loader.lvl).astype(float)

            # revert to original orientation, location, and resolution
            seg[:, 0] *= ds[0]
            seg[:, 1] *= ds[1]

            frame_thresh = Image.open(out_f.replace('.json', '_%d_THRESH.png') % roi_i)

            seg.translate(-crop_loc)
            roi.translate(loc)
            seg = seg.rotate(roi.centroid()[0], -angle)
            seg.translate(-loc)
            loader.converter.rescale(seg, 0)
            print(loc, crop_loc, roi.centroid(), seg.centroid())

            # connect the last point to the first point
            seg = seg.connect()
            segs.append(seg)
        #
        # end of gm_rois loop

        sana_io.write_annotations(out_f, segs, class_name=args.seg_class)
    #
    # end of slides loop
#
# end of main

def get_density(args, loader, out_f, roi_i, roi):
    if os.path.exists(out_f.replace('.json', '_%d_DENSITY.npy') % roi_i) and \
    os.path.exists(out_f.replace('.json', '_%d_DAT.csv') % roi_i):
        frame_density = np.load(out_f.replace('.json', '_%d_DENSITY.npy') % roi_i)
        fp = open(out_f.replace('.json', '_%d_DAT.csv') % roi_i)
        angle, l0, l1, c0, c1, ds0, ds1 = fp.read().split('\n')[0].split(',')
        loc = Point(float(l0), float(l1), False, 0)
        crop_loc = Point(float(c0), float(c1), False, 0)
        ds = Point(float(ds0), float(ds1), False, 0)
        metrics = [float(angle), loc, crop_loc, ds]
    else:
        # pre-calculate the tissue threshold
        print("----> Pre-Calculating the Tissue/Slide Threshold")
        thumbnail = loader.thumbnail.copy()
        tissue_threshold = thumbnail.get_tissue_threshold(blur=5)

        # define the bounding box of the ROI
        c, r = roi.centroid()
        loc = Point(c[0]-r, c[1]-r, False, 0)
        size = Point(2*r, 2*r, False, 0)
        roi.translate(loc)
        t1 = time.time()

        # calculate the angular rotation of the tissue/slide boundary
        print("------> Detecting Tissue Rotation: ", end="")
        try:
            frame_thumb = loader.load_frame(loc, size, lvl=loader.lc-1)
        except:
            print("Load Failed --", slide_f, '--', roi_i)
        converter = loader.converter
        converter.rescale(roi, loader.lc-1)
        angle = frame_thumb.tissue_angle(tissue_threshold, roi, 1e5, 1e5)
        converter.rescale(roi, loader.lvl)
        converter.rescale(loc, loader.lvl)
        converter.rescale(size, loader.lvl)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()
        if angle is None:
            return None, None

        # load the frame based on the radius of the ROI
        print("------> Loading Frame: ", end="", flush=True)
        frame = loader.load_frame(loc, size)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()

        # rotate the frame around the centroid of the ROI
        print("------> Rotating Frame: ", end="", flush=True)
        rot_center = roi.centroid()[0]
        roi = roi.rotate(rot_center, angle)
        frame = frame.rotate(angle)
        crop_loc, crop_size = roi.bounding_box()
        frame = frame.crop(crop_loc, crop_size)
        roi.translate(crop_loc)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()

        # generate the tissue mask from the rotated ROI
        print("------> Detecting Tissue in ROI: ", end="", flush=True)
        tissue_mask = frame.copy()
        tissue_mask.detect_tissue(tissue_threshold, 1e5, 1e5)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()

        # separate the stain from the ROI using color deconvolution
        print("------> Stain Separating: ", end="", flush=True)
        if 'NeuN' in out_f:
            target = 'DAB'
        else:
            target = 'HEM'
        separator = StainSeparator(args.stain, target, ret_od=True)
        frame_stain = separator.run(frame.img)[0]
        frame_stain = Frame(frame_stain, loader.lvl, converter)
        frame_stain.save(out_f.replace('.json', '_%d_STAIN.png') % roi_i)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()

        print("------> Stain Thresholding: ", end="", flush=True)
        stain_threshold = frame_stain.copy().get_stain_threshold(tissue_mask)
        frame_thresh = frame_stain.copy()
        frame_thresh.threshold(stain_threshold, x=0, y=1)
        frame_thresh.save(out_f.replace('.json', '_%d_THRESH.png') % roi_i)
        print("%.3f sec" % (time.time() - t1))
        t1 = time.time()

        print("------> Calculating Tile Features", end="", flush=True)

        # load the tiles
        tsize = Point(args.tsize[0], args.tsize[1], True)
        tstep = Point(args.tstep[0], args.tstep[1], True)
        tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        tiler.set_frame(frame_thresh, pad=True)
        tiles = tiler.load_tiles()
        frame_density = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
        frame_size = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)

        metrics = [angle, loc, crop_loc, tiler.ds]
        fp = open(out_f.replace('.json', '_%d_DAT.csv' % roi_i), 'w')
        fp.write('%f,%f,%f,%f,%f,%f,%f\n' % (angle, loc[0], loc[1], crop_loc[0], crop_loc[1], tiler.ds[0], tiler.ds[1]))
        fp.close()

        for j in range(tiles.shape[1]):
            for i in range(tiles.shape[0]):
                s = ' -- %d/%d:' % (j*tiles.shape[0]+i,
                    tiles.shape[0]*tiles.shape[1])
                print(s + "\b"*len(s), flush=True, end="")
                frame = Frame(tiles[i][j], 0, converter)
                frame.detect_cells(min_body=15, max_body=1000)
                count = len(frame.get_bodies())
                areas = np.array([c.polygon.area() for c in frame.get_bodies()])
                areas = areas[areas > 50]
                if len(areas) < 5:
                    area = 0
                else:
                    area = np.mean(areas)
                frame_density[i][j] = count
                frame_size[i][j] = area
        np.save(out_f.replace('.json', '_%d_DENSITY.npy' % roi_i), frame_density)
        np.save(out_f.replace('.json', '_%d_SIZE.npy' % roi_i), frame_size)
        roi.translate(-crop_loc)
        roi = roi.rotate(rot_center, -angle)
        roi.translate(-loc)
        print("%.3f sec" % (time.time() - t1), flush=True)
        t1 = time.time()
    return frame_density, metrics

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
