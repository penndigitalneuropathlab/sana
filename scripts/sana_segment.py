#!/usr/bin/env python3

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np
import cv2
from scipy.ndimage import convolve1d

# custom modules
import sana_io
from sana_params import Params
from sana_loader import Loader
from sana_tiler import Tiler
from sana_geo import Line

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

def get_01(n, x, v0, v1):
    temp = np.zeros(n)
    temp[:x] = v0
    temp[x:] = v1
    return temp

def get_012(n, x, y, v0, v1, v2):

    flat_percent = 0.6
    incline_percent = (1 - flat_percent) / 2
    xp = int((y-x)*incline_percent) + x
    yp = y - int((y-x)*incline_percent) 
    temp = np.zeros(n)
    temp[:x] = v0
    temp[x:xp] = np.linspace(v0, v1, xp-x)
    temp[xp:yp] = v1
    temp[yp:y] = np.linspace(v1, v2, y-yp)
    temp[y:] = v2
    return temp

def get_ind(sig, v0, v1, l=None, r=None):
    score = np.zeros_like(sig)                
    for x in range(sig.shape[0]):
        temp = get_01(sig.shape[0], x, v0, v1)
        if l is None or r is None:
            dist = 0
        else:
            dist = (np.abs(l-x) + np.abs(r-x)) / 400
        score[x] = np.sum(np.abs(sig-temp)) + dist
    return np.argmin(score)

def get_inds(sig, v0, v1, v2, l=None, r=None):
    score = np.full((sig.shape[0], sig.shape[0]), np.inf)
    dist = np.full((sig.shape[0], sig.shape[0]), 0)
    for x in range(sig.shape[0]):
        for y in range(x+10, sig.shape[0]):
            temp = get_012(sig.shape[0], x, y, v0, v1, v2)
            if l is None or r is None:
                dist[x][y] = 0
            else:
                dist[x][y] = (np.abs(l[0]-x) + np.abs(l[1]-y) + np.abs(r[0]-x) + np.abs(r[1]-x))
            score[x][y] = np.sum(np.abs(sig-temp))
    score /= np.max(score[score!=np.inf])
    orig_score = np.copy(score)
    if not l is None:
        dist = 10*(dist - np.min(dist, axis=0)) / (np.max(dist, axis=0)-np.min(dist, axis=0))
        score *= dist
    if not l is None:
        fig, axs = plt.subplots(1,2)
        axs[0].plot(sig)
        axs[1].plot(score[10])
        axs[1].plot(dist[10])
        axs[1].plot(orig_score[10])                
        plt.show()
    return np.unravel_index(np.argmin(score), score.shape)

def get_l(img, st, en):
    l = np.zeros(img.shape[1])
    for j in range(img.shape[1]):
        sig = img[st:en, j]
        v0 = np.mean(img[st,:])
        v1 = np.mean(img[en,:])
        l[j] = get_ind(sig, v0, v1) + st
    return l

def get_ls(img, st, en):
    l0, l1 = np.zeros(img.shape[1]), np.zeros(img.shape[1])
    for j in range(img.shape[1]):
        sig = img[st:en, j]
        v0 = np.mean(img[st,:])
        #v1 = np.max(sig)
        v1 = np.max(img[st:en, :])
        v2 = np.mean(img[en,:])
        inds = get_inds(sig, v0, v1, v2)
        l0[j] = inds[0] + st
        l1[j] = inds[1] + st
    return l0, l1

# this script loads a series of slides and landmark vectors
# it loads and processes the data near the vector, and generates
# a various segmentation ROI's representing GM, WM and subGM regions
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the slide files to process
    slides = sana_io.get_slides_from_lists(args.lists)
    if len(slides) == 0:
        print("***> No Slides Found")
        parser.print_usage()
        exit()

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):

        # progress messaging
        print('--> Processing Slide: %s (%d/%d)' % \
              (os.path.basename(slide_f), slide_i, len(slides)), flush=True)

        # read the landmark vectors from the annotation file
        
        # NOTE: this is just for validation purposes
        #        will eventually delete!!
        if 'NeuN' in slide_f or 'SMI32' in slide_f or 'SMI94' in slide_f or 'parvalbumin' in slide_f:
            landmarks_f = sana_io.create_filepath(
                slide_f, ext='.npy', fpath=args.adir, rpath=args.rdir)
            try:
                v = np.load(landmarks_f)
            except:
                print('Skipping:', slide_f)
                continue

            # convert the array to a Line array
            v = Line(v[:,0], v[:,1], False, 0)            
        
            layers = None
            # layers_f = sana_io.create_filepath(
            #     slide_f, ext='.json', fpath='annotations/rois')
            # annos = sana_io.read_annotations(layers_f)
            # LAYERS = ['Layer I', 'Layer II', 'Layer III', 'Layer IV', 'Layer V', 'Layer VI']
            # layers = []
            # for layer in LAYERS:
            #     try:
            #         a = [a for a in annos if a.class_name == layer][0]
            #         layers.append(a)
            #     except:
            #         layers.append(None)
            # if any([x is None for x in layers]):
            #     continue
            
        # just load the landmarks that were annotated
        # TODO: clean this up
        else:    
            landmarks_f = sana_io.create_filepath(slide_f, ext='.json', fpath=args.adir)
            if not os.path.exists(landmarks_f):
                continue
            v = sana_io.read_annotations(landmarks_f, class_name='LANDMARKS')
            if len(v) == 0:
                continue
            v = v[0]
            layers = None

     
        # initialize the loader
        try:
            loader = Loader(slide_f)
        except Exception as e:
            print(e)
            print('***> Warning: Could\'t load .svs file, skipping...')
            continue

        # set the image resolution level
        loader.set_lvl(args.lvl)

        # TODO: loop through multiple vectors!

        # initialize the Params IO object, this stores params
        # relating to loading/processing the Frame
        params = Params()
        
        # load the frame into memory
        # TODO: clean this up!
        print('Loading the Frame', flush=True)
        frame, v, layers, M, orig_frame = loader.from_vector(params, v, layers)

        # get the processor object
        processor = sana_io.get_processor(
            slide_f, frame, args.debug)

        # TODO: error checking
        if processor is None:
            continue
        
        # run the GM segmentation algorithm!
        processor.run_segment(args.odir, params)

        exit()
        
        # get the grayscale version
        frame_gray = frame.copy()
        frame_gray.to_gray()
        
        # separate the stains out of the frame
        print('Color Deconvolution', flush=True)
        if 'NeuN' in slide_f:
            #ss = StainSeparator('H-DAB', [True, True, False])
            ss = StainSeparator('H-DAB', 'DAB')            
        else:
            ss = StainSeparator('H-DAB', 'HEM')
        stain = ss.run(frame.img)
        stain = 255 - stain
        frame_stain = Frame(stain, loader.lvl, loader.converter, loader.csf_threshold)

        print('Normalization', flush=True)
        frame_norm = mean_normalize(loader, frame_stain)[0]

        print('Filtering', flush=True)
        frame_filt = frame_norm.copy()
        frame_filt.anisodiff()

        # TODO: try other threshold techniques
        print('Thresholding', flush=True)
        thresh = get_stain_threshold(frame_filt, 5, 255)
        frame_thresh = frame_filt.copy()
        frame_thresh.threshold(thresh, 0, 255)

        print('More Filtering', flush=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cv2.morphologyEx(frame_thresh.img, cv2.MORPH_OPEN, kernel=kernel, dst=frame_thresh.img)

        print('Detecting', flush=True)
        frame_thresh.img = frame_thresh.img // 255
        frame_thresh.get_contours()
        frame_thresh.filter_contours(min_body_area=50)
        detections = frame_thresh.get_body_contours()
        frame_thresh = create_mask([x.polygon for x in detections], frame.size(), frame.lvl, frame.converter)
        
        alpha = 0.5
        plot_img = frame.img.astype(float) / 255
        plot_thresh = frame_thresh.img.astype(float)
        z = np.zeros((frame.img.shape[0], frame.img.shape[1], 1))
        plot_thresh = np.concatenate((plot_thresh, plot_thresh, plot_thresh, plot_thresh-(1-alpha)), axis=-1)

        cell_centers = np.array([np.mean(d.polygon, axis=0) for d in detections])
        cell_sizes = np.array([d.polygon.area() for d in detections])
        cell_ints = np.array([frame_filt.get_intensity(d.polygon) for d in detections])
        
        #cell_ecc = np.array([d.polygon.ecc() for d in detections])

        # TODO: gray_feat shold be calculated with a guass blur with very thin y axis, separate from tiles!!!
        print(frame_gray.img.shape, frame.img.shape)
        gray_feat = cv2.GaussianBlur(frame_gray.img, ksize=(0,0), sigmaX=10, sigmaY=1)
        print(gray_feat.shape)
        
        tsize = Point(400, 100, True)
        tstep = Point(10, 10, True)
        tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        tiler.set_frame(frame_gray)
        ds = tiler.ds
        tpad = tiler.fpad
        tiles = tiler.load_tiles()

        cell_feat = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
        size_feat = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)
        int_feat = np.zeros((tiles.shape[0], tiles.shape[1]), dtype=float)        
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
                # TODO: should actually pad the frame by the tilesize before loading!                 
                x = [tile_loc[0], tile_loc[0]+tile_size[0], tile_loc[0]+tile_size[0], tile_loc[0]]
                y = [tile_loc[1], tile_loc[1], tile_loc[1]+tile_size[1], tile_loc[1]+tile_size[1]]
                if x[0] < 0:
                    x[0] = 0
                    x[3] = 0
                if x[1] > frame.img.shape[1]:
                    x[1] = frame.img.shape[1]
                    x[2] = frame.img.shape[1]                    
                if y[0] < 0:
                    y[0] = 0
                    y[1] = 0
                if y[2] > frame.img.shape[0]:
                    y[2] = frame.img.shape[0]
                    y[3] = frame.img.shape[0]                    
                tile = Polygon(x, y, False, 0)

                # get the neurons that are in this tile

                inds = (
                    (
                        (cell_centers[:,0] > x[0]) & (cell_centers[:,0] < x[1]) & \
                        (cell_centers[:,1] > y[0]) & (cell_centers[:,1] < y[2])
                    ) 
                    & ((cell_sizes > 10) & (cell_sizes < 750))
                )
                

                # TODO: check the math here                                                
                density = 1000 * np.sum(inds) / tile.area()
                if density != 0:
                    avg_size = np.mean(cell_sizes[inds])
                else:
                    avg_size = 0
                #avg_ecc = np.mean(cell_eccs[inds])
                avg_int = np.mean(cell_ints[inds])
                
                cell_feat[i][j] = density
                #gray_feat[i][j] = np.mean(tiles[i][j])
                size_feat[i][j] = avg_size
                int_feat[i][j] = avg_int
                
        # make sure the landmarks fit in the image
        v = np.clip(v.astype(int), 0, gray_feat.shape[0]-1)

        # get the CSF/GM boundary
        csf_gm = get_l(gray_feat, v[0,1], v[1,1])

        # scale the landmarks to the tiled image
        v = (v/tiler.ds).astype(int)
        v = np.clip(v, 0, cell_feat.shape[0]-1)

        # get the L34 and L45 boundary
        l34, l45 = get_ls(cell_feat, v[2,1], v[3,1])

        # get the L12 boundary
        # NOTE: the l23 boundary may not always be accurate (i.e. landmark is placed in L2),
        #         however using this function instead of get_l() should be more consistent for L12
        l12, l23 = get_ls(cell_feat, v[1,1], v[2,1])

        # get the GM/WM boundary
        # NOTE: same reason we use get_ls here as L12
        l56, gm_wm = get_ls(cell_feat, v[3,1], v[4,1])

        # moving average filter the CSF boundary
        Mfilt = 151
        csf_gm = convolve1d(csf_gm, np.ones(Mfilt)/Mfilt, mode='reflect')

        # TODO: this is not really the way to do it, theres too much large noise that the script attaches to
        #        really want to do what paul implemented, the general density of the layer shouldn't change that much
        # moving average filter the layer boundaries
        Nfilt = 21        
        layers = []
        for x in [l12, l34, l45, gm_wm]:
            layers.append(convolve1d(x, np.ones(Nfilt)/Nfilt, mode='reflect'))
                            
        # scale back to original resolution
        l_x = np.arange(cell_feat.shape[1]) * tiler.ds[0]        
        l_y = [x * tiler.ds[1] for x in layers]
        csf_gm_x = np.linspace(0, l_x[-1], gray_feat.shape[1])        
        csf_gm_y = csf_gm

        # generate and store the annotations
        CLASSES = [
            'CSF_BOUNDARY',
            'L1_L2_BOUNDARY',
            'L3_L4_BOUNDARY','L4_L5_BOUNDARY',
            'GM_WM_BOUNDARY'
        ]
        csf = Polygon(csf_gm_x, csf_gm_y, False, LVL).to_annotation(ofname, CLASSES[0])
        a = [csf] + [Polygon(l_x, l_y[i], False, LVL).to_annotation(ofname, CLASSES[i+1]) \
                     for i in range(len(l_y))]

        # plot the feats
        if args.debug:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(cell_feat, cmap='coolwarm')
            ax[0].plot(v[:,0], v[:,1], 'x', color='red')
            ax[0].set_title('Cell Density')
            ax[1].plot(v[:,0], v[:,1], 'x', color='red')
            ax[1].imshow(size_feat, cmap='coolwarm', vmin=80)
            ax[1].plot(v[:,0], v[:,1], 'x', color='red')
            ax[1].set_title('Average Size')
            ax[2].imshow(int_feat, cmap='coolwarm')
            ax[2].plot(v[:,0], v[:,1], 'x', color='red')
            ax[2].set_title('Average Intensity')
            ax[2].imshow(gray_feat, cmap='coolwarm')
            ax[2].plot(v[:,0], v[:,1], 'x', color='red')
            ax[2].set_title('Grayscale Intensity')
            
            # plot the frames
            fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
            ax[0].imshow(frame.img)
            ax[0].set_title('Orig')
            ax[1].imshow(frame_filt.img, cmap='coolwarm', vmin=0, vmax=255)
            ax[1].set_title('Stain Sep. + Filter')
            ax[2].imshow(frame_thresh.img, cmap='coolwarm')
            #ax[2].imshow(plot_thresh)
            ax[2].set_title('Thresh')
            
            fig, ax = plt.subplots(1,1)
            ax.imshow(frame.img)
            for x in a:
                plot_poly(ax, x, color='blue', last=False)

            plt.show()
            
        # perform the inverse transform
        for i in range(len(a)):
            a[i].translate(-writer.data['crop_loc'])
            b = a[i].transform_inv(M)
            a[i] = b.to_annotation(a[i].file_name, a[i].class_name, a[i].name, a[i].confidence)
            a[i].translate(-writer.data['loc'])

        # finally, write the predictions
        sana_io.write_annotations(ofname, a)

        # # plot the original slide
        # fig, ax = plt.subplots(1,1)
        # ax.imshow(orig_frame.img)
        # for x in a:
        #     x.translate(writer.data['loc'])
        #     plot_poly(ax, x)
        #     x.translate(-writer.data['loc'])
            
def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True)
    parser.add_argument('-adir', type=str, default="",
                        help="specify the location of annotation files\n[default: same as slide]")
    parser.add_argument('-odir', type=str, default="",
                        help="specify the location to write files to\n[default: same as slide]")
    parser.add_argument('-rdir', type=str, default="",
                        help="specify directory path to replace\n[default: ""]")
    parser.add_argument('-lvl', type=int, default=0,
                        help='pixel resolution to use, 0 is maximum.')
    parser.add_argument('-debug', action='store_true')
    return parser
#
# end of cmdl_parser

if __name__ == '__main__':
    main(sys.argv)

#
# end of file


