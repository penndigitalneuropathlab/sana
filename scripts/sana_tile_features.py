#!/usr/bin/env python

# system modules
import os
import sys
import time
import argparse

# installed modules
import numpy as np
from matplotlib import pyplot as plt

# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, mean_normalize, create_mask, get_stain_threshold
from sana_geo import Array, Point, Polygon, plot_poly
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader
from sana_thresholds import knn

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
        print('--> Processing Slide: %s (%d/%d)' % (slide_f, slide_i, len(slides)))
        
        # get the annotation file
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)

        # load the rois
        rois = sana_io.read_annotations(anno_f, class_name=args.roi_class)
        
        # loop through the frames to process
        for roi_i, roi in enumerate(rois):
            print('----> Processing Frame (%d/%d)' % \
                  (roi_i+1, len(rois)), flush=True)

            # get the output file names
            density_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='_%d_DENSITY' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            size_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='_%d_SIZE' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            
            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            writer = DataWriter(data_f)
            lvl = writer.data['lvl']
                        
            # initialize the loader
            loader = Loader(slide_f)
            loader.set_lvl(lvl)
            converter = loader.converter

            # load the tissue mask
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            if not os.path.exists(mask_f):
                continue
            frame_tissue = Frame(mask_f, lvl, converter)
            
            # load the detections
            anno_f = sana_io.create_filepath(slide_f, ext='.json',
                                             fpath=args.idir, rpath=args.rdir)
            cells = sana_io.read_annotations(anno_f, class_name=args.detection_class)
            [x.translate(writer.data['loc']+writer.data['crop_loc']) for x in cells]

            # rotate the data by the angle of the tissue boundary
            [x.rotate(frame_tissue.size()//2, writer.data['angle']) for x in cells]

            # fig, axs = plt.subplots(1,1)
            # axs.imshow(frame_tissue.img)
            # for x in cells:
            #     plot_poly(axs, x, color='red')
            # plt.show()
            # exit()
            
            # load the tiles in the frame
            # TODO: this tiler needs to be restricted to inside the segmentation
            #       this means the center aligned tile centers will be 0 + tile_size//2
            # TODO: layer 6 should allow a padding into the WM
            tsize = Point(args.tsize[0], args.tsize[1], True)
            tstep = Point(args.tstep[0], args.tstep[1], True)
            tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
            tiler.set_frame(frame_tissue)
            ds = tiler.ds
            writer.data['ds'] = ds
            tpad = tiler.fpad
            tiles = tiler.load_tiles()
            
            # prepare the tile feature matrices
            frame_density = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
            frame_size = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
            
            # loop through the tiles
            for j in range(tiles.shape[1]):
                for i in range(tiles.shape[0]):
                    s = '%d/%d' % (i+j*tiles.shape[0], (tiles.shape[0]*tiles.shape[1]))
                    print(s+'\b'*len(s),end="",flush=True)
                    
                    # get the current tile's loc and size
                    tile_loc = Point(j * tiler.step[1], i * tiler.step[0], False, 0)
                    tile_size = tiler.size

                    # center align the tile
                    tile_loc -= tile_size//2

                    # build the tile polygon
                    x = [tile_loc[0], tile_loc[0]+tile_size[0], tile_loc[0]+tile_size[0], tile_loc[0]]
                    y = [tile_loc[1], tile_loc[1], tile_loc[1]+tile_size[1], tile_loc[1]+tile_size[1]]
                    tile = Polygon(x, y, False, 0)

                    # get the neurons that are in this tile
                    local_cells = [x for x in cells if x.inside(tile)]
                    
                    # calculate the density and avg area of the neurons
                    # TODO: check the math here
                    density = 1000 * len(local_cells) / (tsize[0] * tsize[1])
                    if len(local_cells) == 0 or True:
                        avg_area = 0
                    else:
                        avg_area = np.mean([x.area() for x in local_cells])

                    frame_density[i][j] = density
                    frame_size[i][j] = avg_area
                #
                # end of i tiles loop
            #
            # end of j tiles loop

            # finally, save the tile metrics
            writer.write_data()
            np.save(density_f, frame_density)
            np.save(size_f, frame_size)
        #
        # end of rois loop
    #
    # end of slides loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True)
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

if __name__ == '__main__':
    main(sys.argv)

#
# end of file
