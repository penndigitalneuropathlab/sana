#!/usr/bin/env python

# system modules
import os
import sys
import time
import argparse

# installed modules
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import pyplot as plt

# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, mean_normalize, create_mask, get_stain_threshold
from sana_geo import Array, Point, Polygon
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader
from sana_thresholds import knn

DEF_NAME_ROI = '_ORIG'
DEF_NAME_STAIN = '_STAIN'
DEF_NAME_THRESH = '_THRESH'

def main(argv):
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
        print(slide_f)
        # get the annotation file
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # get the output annotation file
        out_f = sana_io.create_filepath(
            slide_f, ext='_SEG.json', fpath=args.idir, rpath=args.rdir)

        # load the frame and get the rotation angle
        roi_f = sana_io.create_filepath(slide_f, ext='.json',
                                        fpath=args.adir, rpath=args.rdir)
        rois = sana_io.read_annotations(roi_f, class_name=args.roi_class)
        
        # loop through the frames to process
        segs = []
        for roi_i, roi in enumerate(rois):
            print('----> Processing Frame (%d/%d)' % \
                  (roi_i+1, len(rois)), flush=True)

            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            writer = DataWriter(data_f)
            lvl = writer.data['lvl']
            loc = writer.data['loc']
            size = writer.data['size']

            roi.translate(loc)
                        
            # initialize the loader
            loader = Loader(slide_f)
            loader.set_lvl(lvl)
            converter = loader.converter
            
            frame = loader.load_frame(loc, size, lvl=1)
            converter.rescale(loc, lvl)
            angle = frame.get_rotation(roi)
            writer.data['tissue_angle'] = angle
            
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            frame_tissue = Frame(mask_f, lvl, converter)
            
            # load the detections
            anno_f = sana_io.create_filepath(slide_f, ext='.json',
                                             fpath=args.idir, rpath=args.rdir)
            cells = sana_io.read_annotations(anno_f, class_name=args.detection_class)
            [x.translate(loc) for x in cells]

            # rotate the data by the angle of the tissue boundary
            center = roi.centroid()[0]
            [x.rotate(center, angle) for x in cells]
            roi.rotate(center, angle)
            frame_tissue.rotate(angle)

            # crop the data to fit the rotated roi
            crop_loc, crop_size = roi.bounding_box()            
            [x.translate(crop_loc) for x in cells]            
            roi.translate(crop_loc)
            frame_tissue.crop(crop_loc, crop_size)
            
            # load the tiles in the frame
            tsize = Point(args.tsize[0], args.tsize[1], True)
            tstep = Point(args.tstep[0], args.tstep[1], True)
            tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
            tiler.set_frame(frame_tissue)
            ds = tiler.ds
            tpad = tiler.fpad
            tiles = tiler.load_tiles()
            
            # prepare the tile feature matrices
            frame_density = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
            frame_size = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)

            density_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='DENSITY_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            size_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='SIZE_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)            
            frame_density = np.load(density_f)
            frame_size = np.load(size_f)
            
            # # loop through the tiles
            # for j in range(tiles.shape[1]):
            #     for i in range(tiles.shape[0]):
            #         s = '%d/%d' % (i+j*tiles.shape[0], (tiles.shape[0]*tiles.shape[1]))
            #         print(s+'\b'*len(s),end="",flush=True)
                    
            #         # get the current tile
            #         tile_loc = Point(j * tiler.step[1], i * tiler.step[0], False, 0)
            #         tile_size = tiler.size
            #         x = [tile_loc[0], tile_loc[0]+tile_size[0], tile_loc[0]+tile_size[0], tile_loc[0]]
            #         y = [tile_loc[1], tile_loc[1], tile_loc[1]+tile_size[1], tile_loc[1]+tile_size[1]]
            #         tile = Polygon(x, y, False, 0)

            #         # get the neurons that are in this tile
            #         local_cells = [x for x in cells if x.inside(tile)]
                    
            #         # calculate the density and avg area of the neurons
            #         # TODO: check the math here
            #         density = 1000 * len(local_cells) / (tsize[0] * tsize[1])
            #         if len(local_cells) == 0 or True:
            #             avg_area = 0
            #         else:
            #             avg_area = np.mean([x.area() for x in local_cells])

            #         frame_density[i][j] = density
            #         frame_size[i][j] = avg_area
            #     #
            #     # end of i tiles loop
            # #
            # # end of j tiles loop

            # # finally, save the tile metrics
            # np.save(density_f, frame_density)
            # np.save(size_f, frame_size)
            
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(frame_density)
            axs[1].imshow(frame_size)
            plt.show()

            # apply KNN to get the 3 most prominent values
            # NOTE: in DAB this will be Slide/WM, low GM, and high GM
            # NOTE: in HEM this will be Slide, GM, and WM
            try:
                seg, thresholds = knn(frame_density, 3)
            except:
                continue
            d_seg = seg
            seg = Frame(seg)
            
            if 'NeuN' in slide_f:
                seg.threshold(thresholds[0], x=0, y=1)
                seg = seg.img[:, :, 0]                
            else:
                seg.threshold(thresholds[1], x=1, y=0)
                seg = seg.img[:, :, 0]

            plt.imshow(seg)
            plt.show()
                
            # define the min thickness for layer 0 and layer 6 in microns
            N0 = Array([10.], is_micron=True)
            converter.to_pixels(N0, loader.lvl)
            N6 = Array([100.], is_micron=True)
            converter.to_pixels(N6, loader.lvl)
            N6 = N6 / ds[1]
            N0, N6 = int(N0[0]), int(N6[0])
            l0, l6 = [], []
            
            # loop from bottom to top in each column, find the first instance
            # of positive thresholded data
            for i in range(seg.shape[1]):
                x = seg[:, i]
                found = False
                wm_started = False                
                for j in range(x.shape[0]-1, -1, -1):
                    if all(x[j-N0:j] == 0):
                        wm_started = True
                    if all(x[j-N6:j] == 1) and wm_started:
                        l6.append(j)
                        found = True
                        break
                if not found:
                    l6.append(x.shape[0]-1)
            l6 = np.array(l6)

            # remove the WM using the detected boundary
            for i in range(frame_density.shape[1]):
                frame_density[l6[i]:, i] = 0
                
            plt.imshow(frame_density)
            plt.show()
            
            # if we're using HEM stain, need to re-run knn
            # to get slide, low GM, and high GM
            if 'NeuN' not in slide_f:
                try:
                    seg, thresholds = knn(frame_density, 3)
                except:
                    continue
                plt.imshow(seg)
                plt.show()
            
                seg = Frame(seg)
                seg.threshold(thresholds[0], x=0, y=1)
                seg = seg.img[:, :, 0]
            plt.imshow(seg)
            plt.show()
            
            # loop from bottom to top of each column, find the first
            # instance of zeroed slide
            for i in range(0, frame_tissue.img.shape[1], 20):
                x = frame_tissue.img[:, i]
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

            l0 = uniform_filter1d(l0, x.shape[0]//16)
            l6 = uniform_filter1d(l6, x.shape[0]//4)
            # build the gm segmentation by combining layers 0 and 6
            x = np.concatenate([x[::-1], x])
            y = np.concatenate([l0[::-1], l6])
            seg = Polygon(x, y, False, loader.lvl).astype(float)

            # revert from the tile dimension
            seg[:, 0] *= ds[0]
            seg[:, 1] *= ds[1]

            print(seg.shape)
            # # measure the linearity and parallelness
            # linearity = get_linearity(seg)
            # thickness = get_thickness(seg)

            # deviation = 150 / loader.mpp
            # mu, sigma = np.mean(thickness), np.std(thickness)
            # mid = thickness.shape[0]//2
            # left = np.argmin(np.abs(thickness[:mid] - thickness[mid]) > deviation)
            # right = mid + np.argmax(np.abs(thickness[mid:] - thickness[mid]) > deviation)

            # seg_left = seg[seg.shape[0]//2+left, 0]
            # seg_right = seg[seg.shape[0]//2+right, 0]
            # new_seg = seg.copy()
            # new_seg = new_seg[np.where(new_seg[:, 0] < seg_right)]
            # new_seg = new_seg[np.where(new_seg[:, 0] > seg_left)]
            # new_seg = new_seg.connect()

            seg.rotate(frame_tissue.size()//2, -angle)

            # remove vertices that are not in the roi
            seg = seg.filter(roi)

            # translate to slide origin
            seg.translate(-(loc+crop_loc))

            # convert to max resolution
            loader.converter.rescale(seg, 0)

            # connect the last point to the first point
            seg = seg.connect()

            # finally, store the detected segmentation
            segs.append(seg)

            print('')
        #
        # end of crude ROI loop

        # convert the detections to Annotations
        annos = []
        for seg in segs:
            anno = seg.to_annotation(slide_f, args.seg_class)
            annos.append(anno)

        # write the annotations to output annotation file
        sana_io.write_annotations(out_f, annos)
    #
    # end of slides loop
#
# end of main

# TODO: implement this! is it actually needed?
def get_linearity(seg):
    l0, l6 = separate_seg(seg)

    return None

def get_thickness(seg):
    l0, l6 = separate_seg(seg)

    thickness = l6[:, 1] - l0[:, 1]
    mu, sigma = np.mean(thickness), np.std(thickness)
    parallelness = (thickness - mu) / sigma
    return thickness

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*')
    parser.add_argument('-seg_class', type=str, default='GM_SEG',
                        help="class name of the GM Segmentation")
    parser.add_argument('-roi_class', type=str, default='GM_ROI',
                        help="class name of the GM Crude ROI")
    parser.add_argument('-detection_class', type=str, default='CELL',
                        help='class name of the detected object to process')
    parser.add_argument('-tsize', type=int, nargs=2, default=(500, 300),
                        help="tile size for analysis\n[default: 500 300]")
    parser.add_argument('-tstep', type=int, nargs=2, default=(25, 25),
                        help="tile step for analysis\n[default: 25 25]")
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-idir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
