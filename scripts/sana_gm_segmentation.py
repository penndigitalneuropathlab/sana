<<<<<<< HEAD
filt#!/usr/local/bin/python3.9
=======
#!/usr/bin/env python
>>>>>>> 15520819e9b287758735d8743a23c2de8f71e6cc

# system modules
import os
import sys
import argparse

# installed modules
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import pyplot as plt

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, get_stain_threshold
from sana_geo import Array, Point, Polygon
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

        # get the output annotation file
        out_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.odir, rpath=args.rdir)

        # define the target stain to process based on the slide filename
        if 'NeuN' in slide_f:
            target = 'DAB'
        else:
            target = 'HEM'

        # initialize the slide loader, set the level
        # try:
        #     loader = Loader(slide_f)
        # except Exception as e:
        #     print(e)
        #     print("Slide not found:", slide_f)
        #     continue
        loader = Loader(slide_f)
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
        for roi_i, roi in enumerate(rois):
            print('----> Processing ROI %d/%d' % \
                  (roi_i, len(rois)), flush=True, end="")

            # get the roi metrics file
            metrics_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d_DAT' % roi_i,
                fpath=args.odir, rpath=args.rdir)

            # get the tissue mask filename
            tissue_f = sana_io.create_filepath(
                out_f, ext='.npy', suffix='_%d_TISSUE' % roi_i)

            filt_f = sana_io.create_filepath(
                out_f, ext='.npy', suffix='_%d_FILT' % roi_i)

            # get the tiled density measurements
            frame_density, frame_size, roi, angle, loc, crop_loc, ds = \
                get_tile_features(args, loader, out_f, metrics_f, roi_i, roi)

            continue

            # get the tissue mask
            frame_tissue = Frame(
                np.load(tissue_f), loader.lvl, loader.converter)

            # apply KNN to get the 3 most prominent values
            # NOTE: in DAB this will be Slide/WM, low GM, and high GM
            # NOTE: in HEM this will be Slide, GM, and WM
            try:
                seg, thresholds = knn(frame_density, 3)
            except:
                continue
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
            N6 = Array([400.], is_micron=True)
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
                try:
                    seg, thresholds = knn(frame_density, 3)
                except:
                    continue
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

            l0 = uniform_filter1d(l0, x.shape[0]//16)
            l6 = uniform_filter1d(l6, x.shape[0]//4)
            # build the gm segmentation by combining layers 0 and 6
            x = np.concatenate([x[::-1], x])
            y = np.concatenate([l0[::-1], l6])
            seg = Polygon(x, y, False, loader.lvl).astype(float)

            # revert from the tile dimension
            seg[:, 0] *= ds[0]
            seg[:, 1] *= ds[1]

            # measure the linearity and parallelness
            linearity = get_linearity(seg)
            thickness = get_thickness(seg)

            # get the tissue mask
            frame_filt = Frame(
                np.load(filt_f), loader.lvl, loader.converter)

            deviation = 150 / loader.mpp
            mu, sigma = np.mean(thickness), np.std(thickness)
            mid = thickness.shape[0]//2
            left = np.argmin(np.abs(thickness[:mid] - thickness[mid]) > deviation)
            right = mid + np.argmax(np.abs(thickness[mid:] - thickness[mid]) > deviation)

            seg_left = seg[seg.shape[0]//2+left, 0]
            seg_right = seg[seg.shape[0]//2+right, 0]
            new_seg = seg.copy()
            new_seg = new_seg[np.where(new_seg[:, 0] < seg_right)]
            new_seg = new_seg[np.where(new_seg[:, 0] > seg_left)]
            new_seg = new_seg.connect()
            fig, axs = plt.subplots(2,1)
            axs[0].imshow(frame_filt.img, cmap='coolwarm')

            # l0, l6 = separate_seg(seg)
            # axs[0].plot(l0, color='red')
            # axs[0].plot(l6, color='blue')
            for i in range(new_seg.shape[0]-1):
                axs[0].plot([new_seg[i][0], new_seg[i+1][0]], [new_seg[i][1], new_seg[i+1][1]], color='black')
            for i in range(seg.shape[0]-1):
                axs[0].plot([seg[i][0], seg[i+1][0]], [seg[i][1], seg[i+1][1]], color='black')

            axs[1].axhline(thickness[mid], color='black')
            axs[1].axhline(thickness[mid] + deviation, color='gray')
            axs[1].axhline(thickness[mid] - deviation, color='gray')
            axs[1].plot(thickness)
            axs[1].axvline(left, color='red')
            axs[1].axvline(right, color='blue')
            plt.show()

            # translate to origin before cropping
            seg.translate(-crop_loc)

            # rotate back to original orientation
            roi.translate(loc)
            seg.rotate(roi.centroid()[0], -angle)

            # remove vertices that are not in the roi
            seg = seg.filter(roi)

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

def separate_seg(x):
    dist = []
    for i in range(x.shape[0]-2):
        a, b = x[i], x[i+1]
        dist.append((a[0] - b[0])**2 + (a[1] - b[1])**2)
    sep = np.argmax(dist)
    l0, l6 = x[:sep], x[sep+1:-1]
    return [l0, l6]

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

def get_tile_features(args, loader, out_f, metrics_f, roi_i, roi):

    # define file location for the density file and roi metrics
    density_f = sana_io.create_filepath(
        out_f, ext='.npy', suffix='_%d_DENSITY' % roi_i)
    size_f = sana_io.create_filepath(
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
        frame_orig, roi, angle, loc, crop_loc = get_stain_separated_frame(
            args, loader, out_f, metrics_f, roi_i, roi)

        # load the tiles in the frame
        tsize = Point(args.tsize[0], args.tsize[1], True)
        tstep = Point(args.tstep[0], args.tstep[1], True)
        tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        tiler.set_frame(frame_filt)
        ds = tiler.ds
        tpad = tiler.fpad

        tiles = tiler.load_tiles()
        orig_tiler = Tiler(loader.lvl, loader.converter, tsize, tstep)
        orig_tiler.set_frame(frame_orig)
        orig_tiles = orig_tiler.load_tiles()

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
        neurons = []
        for j in range(tiles.shape[1]):
            for i in range(tiles.shape[0]):

                # create a frame for the tile
                s = ' -- %d/%d' % (j*tiles.shape[0]+i,
                    tiles.shape[0]*tiles.shape[1])
                print(s + "\b"*len(s), flush=True, end="")

                # detect the centers of each of the cells
                frame = Frame(tiles[i][j], loader.lvl, loader.converter)
                tile_loc = Point(j * tiler.step[1], i * tiler.step[0], False, 0)

                plot = False
                centers = frame.detect_cell_centers(radius, gap, plot)

                # calculate the size of the cells based on the centers
                frame = Frame(tiles[i][j], loader.lvl, loader.converter)
                sizes = frame.detect_cell_sizes(centers, plot)

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

                for k in range(len(centers)):
                    if sizes[k] != 0:
                        r = sizes[k]
                        step = 30
                        c = centers[k]
                        x, y = [], []
                        for theta in range(0, 360+step, step):
                            th = theta * np.pi / 180
                            x.append(c[0] + r * np.cos(th))
                            y.append(c[1] + r * np.sin(th))
                        polygon = Polygon(x, y, False, loader.lvl)

                        polygon.translate(-tile_loc)
                        polygon.translate(tpad//2)
                        neurons.append(polygon)
                #
                # end of centers loop
            #
            # end of i tiles loop
        #
        # end of j tiles loop



        # valid = []
        # for n in neurons:
        #     c = n.centroid()[0]
        #     print(c)
        #     if c[0] < 0 or c[1] < 0:
        #         continue
        #     if c[0] > frame_filt.img.shape[0] or c[1] > frame_filt.img.shape[0]:
        #         continue
        #     valid.append(n)
        n = neurons
        # plt.imshow(frame_filt.img)
        # for p in n:
        #     for x in range(p.shape[0]-1):
        #         plt.plot([p[x][0], p[x+1][0]], [p[x][1], p[x+1][1]], color='red')
        # plt.show()
        # exit()
        roi.translate(loc)
        rot_center = roi.centroid()[0]
        roi.translate(-loc)
        for i in range(len(neurons)):
            n[i].translate(-crop_loc)
            n[i].rotate(rot_center, -angle)
            n[i].translate(-loc)
            n[i] = n[i].connect()
        neurons = n

        neurons = [n for n in neurons if n.filter(roi).shape == n.shape]
        annos = []
        for n in neurons:
            anno = n.to_annotation(loader.fname, 'Neuron')
            annos.append(anno)
        out_neurons_f = density_f.replace('_DENSITY.npy', '_NEURONS.json')
        print('\n\n', len(neurons))
        class_names = ['Neuron'] * len(neurons)
        sana_io.write_annotations(out_neurons_f, annos)

                # if (j > 3):
                #     fig, axs = plt.subplots(1,2)
                #     img = orig_tiles[i][j]
                #     axs[0].imshow(img, cmap='gray')
                #     circles = [plt.Circle(centers[x], sizes[x], color='red', fill=False) for x in range(len(centers)) if sizes[x] != 0]
                #     axs[1].imshow(frame.img)
                #     for c in circles:
                #         axs[0].add_patch(c)
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
    filt_f = sana_io.create_filepath(
        out_f, ext='.npy', suffix='_%d_FILT' % roi_i)

    # check whether we need to generate the frame
    if os.path.exists(filt_f) and not any([args.proc_filt, args.proc_stain]):
        frame_filt = Frame(np.load(filt_f), loader.lvl, loader.converter)
        angle, loc, crop_loc, _, _, _ = sana_io.read_metrics_file(metrics_f)
    else:

        # get the stain separated frame
        frame_stain, roi, angle, loc, crop_loc = get_stain_separated_frame(
            args, loader, out_f, metrics_f, roi_i, roi)

        # normalize the image by the mean to remove any inconsistent background staining
        frame_stain = mean_normalize(loader, frame_stain)

        # anistrophic diffusion filter to smooth cell interiors
        frame_filt = frame_stain.copy()

        # threshold for the cells
        stain_threshold = get_stain_threshold(frame_filt)
        frame_filt.threshold(stain_threshold, x=0, y=255)
        sana_io.write_metrics_file(metrics_f, stain_threshold=stain_threshold)

        # morphological filtering to clean up the background
        # kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # frame_filt.img = cv2.erode(frame_filt.img, kern)[:, :, None]

in

    return frame_filt, roi, angle, loc, crop_loc

def get_stain_separated_frame(args, loader, out_f, metrics_f, roi_i, roi):

    # define file location for the stain separated frame and roi metrics
    stain_f = sana_io.create_filepath(
        out_f, ext='.npy', suffix='_%d_STAIN' % roi_i)
    tissue_f = sana_io.create_filepath(
        out_f, ext='.npy', suffix='_%d_TISSUE' % roi_i)

    # check whether we need to generate the stain separated frame
    if os.path.exists(stain_f) and not any([args.proc_stain]):
        frame_stain = Frame(np.load(stain_f), loader.lvl, loader.converter)
        angle, loc, crop_loc, _, _, _ = sana_io.read_metrics_file(metrics_f)
    else:
        sana_io.write_metrics_file(metrics_f, tissue_threshold=loader.csf_threshold)

        # define the bounding box of the ROI, translate to local origin
        loc, size = roi.bounding_centroid()
        roi.translate(loc)
        sana_io.write_metrics_file(metrics_f, loc=loc)

        # calculate the angular rotation of the tissue/slide boundary
        frame_thumb = loader.load_frame(loc, size, lvl=loader.lc-1)
        loader.converter.rescale(roi, loader.lc-1)
        angle = frame_thumb.get_rotation(roi)
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
        roi.rotate(rot_center, angle)
        frame.rotate(angle)

        # crop the rotated frame to remove unneeded data
        crop_loc, crop_size = roi.bounding_box()
        frame.crop(crop_loc, crop_size)
        roi.translate(crop_loc)
        sana_io.write_metrics_file(metrics_f, crop_loc=crop_loc)

        # separate the stain from the ROI using color deconvolution
        if 'NeuN' in out_f:
            target = 'DAB'
        else:
            target = 'HEM'

        separator = StainSeparator(args.stain, target, ret_od=False)
        frame_stain = separator.run(frame.img)[0]
        frame_stain = Frame(frame_stain, loader.lvl, loader.converter)
        frame_stain.to_gray()
        frame_stain.round()

        # inverse image to create a cell prob. heatmap
        frame_stain.img = 255 - frame_stain.img

        # generate the tissue mask from the rotated, binarized Frame
        frame.get_tissue_contours(1e5, 1e5)
        contours = frame.get_body_contours()
        tissue_mask = create_mask(contours, frame.size(),
                                  frame.lvl, frame.converter)

        # mask out all slide background
        frame_stain.mask(tissue_mask, 0)

        # finally, save the frame
        np.save(tissue_f, tissue_mask.img)
        np.save(stain_f, frame_stain.img)
        roi.translate(-crop_loc)
        roi.rotate(rot_center, -angle)
        roi.translate(-loc)
    return frame_stain, roi, angle, loc, crop_loc

def cmdl_parser(argv):
    parser = argparse.ArgumentParser(usage=open(USAGE).read())
    parser.add_argument('-lists', type=str, nargs='*')
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
