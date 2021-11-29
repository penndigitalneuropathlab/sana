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
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, mean_normalize, create_mask, get_stain_threshold
from sana_geo import Array, Point, Polygon, plot_poly, separate_seg
from sana_color_deconvolution import StainSeparator
from sana_tiler import Tiler
from sana_loader import Loader, get_converter
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
            profile_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='_%d_PROFILE' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            lprofile_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='_%d_LPROFILE' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            
            # load the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            writer = DataWriter(data_f)
            lvl = writer.data['lvl']

            # load the density image
            density_f = sana_io.create_filepath(
                slide_f, ext='.npy', suffix='_%d_DENSITY' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            if not os.path.exists(density_f):
                continue
            density = np.load(density_f)

            # initialize the converter
            converter = get_converter(slide_f)

            # load the tissue mask
            mask_f = sana_io.create_filepath(
                slide_f, ext='.png', suffix='_%d_MASK' % roi_i,
                fpath=args.idir, rpath=args.rdir)
            if not os.path.exists(mask_f):
                continue
            frame_tissue = Frame(mask_f, lvl, converter)
            
            frame_density = Frame(density, lvl, converter)
            converter.rescale(roi, lvl)
            roi.translate(writer.data['loc'] + writer.data['crop_loc'])
            roi.rotate(frame_tissue.size()//2, writer.data['angle'])            
            roi[:,0] /= writer.data['ds'][1]
            roi[:,1] /= writer.data['ds'][0]

            # get the 2 segmentations to process between
            s0, s1 = separate_seg(roi)

            try:
                dmat = deform_heatmap(density, s0, s1, plot=False)
            except:
                continue
            
            # average over x axis to get GM profile
            profile = np.mean(dmat, axis=1)

            # do the same process for each  layer annotation to get a layer 4 profile and layer 5 profile
            # NOTE: only do this if the annotations exist!
            # TODO: store the layer profiels in a single file (100, 6) array instead of (100, 1) array
            lprofiles = []
            if len(sana_io.read_annotations(anno_f, class_name='Layer I')) == 1:

                # load the layer segmentations, along with the overall segmentation
                layers = ['Layer %s' % i for i in ['I', 'II', 'III', 'IV', 'V', 'VI']]
                ls = []
                for layer in layers:
                    s = sana_io.read_annotations(anno_f, class_name=layer)[0]
                    converter.rescale(s, lvl)
                    s.translate(writer.data['loc'] + writer.data['crop_loc'])
                    s.rotate(frame_tissue.size()//2, writer.data['angle'])
                    s[:,0] /= writer.data['ds'][1]
                    s[:,1] /= writer.data['ds'][0]
                    ls.append(s)
                    
                ls = annos_to_boundaries(roi, ls, 0)

                for s0, s1 in zip(ls[:-1], ls[1:]):
                    try:
                        dmat = deform_heatmap(density, s0, s1, plot=False)
                    except:
                        break

                    lprofile = np.mean(dmat, axis=1)
                    
                    lprofiles.append(lprofile)

            # finally, save the profiles
            np.save(profile_f, profile)
            if len(lprofiles) == 6:
                lprofiles = np.array(lprofiles)
                np.save(lprofile_f, lprofiles)
            
            # fig = plt.figure(constrained_layout=True)
            # gs = gridspec.GridSpec(3, 3, figure=fig)
            # ax = fig.add_subplot(gs[0, :])
            # ax.plot(profile)
            # bounds = np.min(profile), np.max(profile)
            # ax.set_ylim(bounds)
            # for i in range(2):
            #     for j in range(3):
            #         ax = fig.add_subplot(gs[i+1, j])
            #         ax.plot(lprofiles[3*i+j])
            #         ax.set_ylim(bounds)
            # plt.show()
        #
        # end of rois loop
    #
    # end of slides loop
#
# end of main

def deform_heatmap(density, s0, s1, plot=False):
            
    # stretch the segmentations to the heatmap indices
    f0 = interp1d(s0[:,0], s0[:,1])
    f1 = interp1d(s1[:,0], s1[:,1])            
    x0 = np.arange(np.ceil(np.min(s0[:,0])), np.floor(np.max(s0[:,0])), dtype=int)
    x1 = np.arange(np.ceil(np.min(s1[:,0])), np.floor(np.max(s1[:,0])), dtype=int)
    y0 = f0(x0)
    y1 = f1(x1)

    # get the x values to process
    x_min = max(np.min(x0), np.min(x1))
    x_max = min(np.max(x0), np.max(x1))
    if x_min < 0:
        x_min = 0
    if x_max > density.shape[1]:
        x_max = density.shape[1]
        
    # setup matrix to fit the stretched profiles
    N = 500
    dmat = np.zeros((N, x_max-x_min))
    for i in range(x_min, x_max):

        # get the thickness bounds from the segmentations
        ind0 = int(y0[i-x_min])
        ind1 = int(y1[i-x_min])
        if ind0 > ind1:
            temp = ind0
            ind0 = ind1
            ind1 = temp
        if ind0 < 0:
            ind0 = 0
        if ind1 > density.shape[0]:
            ind1 = density.shape[0]
                
        # extract the density profile, stretch to fit in matrix
        try:
            d = density[ind0:ind1,i]
            d = signal.resample(d, N)
            dmat[:, i-x_min] = d
        except:
            print(density.shape, ind0, ind1, i, x_min, x_max)
            exit()
    #
    # end of x loop

    if plot:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(density, cmap='coolwarm')
        plot_poly(axs[0], s0)
        plot_poly(axs[0], s1)
        axs[1].imshow(dmat, aspect=0.1, cmap='coolwarm')
        plt.show()

    return dmat
#
# end of deform_heatmap

def annos_to_boundaries(gm, layers, lvl):

    # loop through the layer annotations
    # NOTE: we go from -1 to len so we need to fill in gm on both sides of 1-6
    ls = []
    for i in range(-1, len(layers), 1):
        ls.append(([], []))
        
        # handle edge cases
        if i == -1:
            x = gm
        else:
            x = layers[i]
        if i == len(layers)-1:
            y = gm
        else:
            y = layers[i+1]
            
        # loop through the vertices in current anno
        for j in range(x.shape[0]):
            found = False
            
            # loop through vertices in the next anno
            dists = []
            for z in range(y.shape[0]):

                # TODO: might wanna do within 2 pixels or something
                # find overlapping vertices
                if int(x[j][0]) == int(y[z][0]) and \
                   int(x[j][1]) == int(y[z][1]):
                    found = True
                    break
            #
            # end of y loop
                
            # store overlapping vertices
            if found:
                ls[-1][0].append(x[j][0])
                ls[-1][1].append(x[j][1])
        #
        # end of x loop
            
        # create the Polygon out of the vertices
        ls[-1] = Polygon(ls[-1][0], ls[-1][1], False, lvl)
    #
    # end of layers loop
        
    return ls
#
# end of annos_to_boundaries

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True)
    parser.add_argument('-roi_class', type=str, default='GM_ROI',
                        help="class name of the GM Crude ROI")
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
