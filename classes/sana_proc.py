
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from scipy import signal

import sana_io
import sana_geo
from sana_loader import Loader
from sana_color_deconvolution import StainSeparator
from sana_thresholder import TissueThresholder, CellThresholder
from sana_detector import TissueDetector
from sana_framer import Framer
from sana_tiler import Tiler
from sana_frame import Frame

# TODO: need to optimize this - shouldn't need to use gmm()
def get_tissue_threshold(filename):

    # initialize the Loader
    loader = Loader(filename)
    loader.set_lvl(loader.lc-1)

    # calculate the tissue masking threshold
    tissue_frame = copy(loader.thumbnail)
    thresholder = TissueThresholder(tissue_frame, blur=5)
    thresholder.mask_frame()
    tissue_threshold = thresholder.tissue_threshold

    return tissue_threshold
#
# end of get_tissue_threshold

def segment_gm(args, filename, anno, tissue_threshold, gm_threshold=0.2, count=0):
    loader = Loader(filename)
    loader.set_lvl(args.level)
    v = anno.vertices()
    for i in range(len(v)-1):
        plt.plot([v[i][0], v[i+1][0]], [v[i][1], v[i+1][1]], color='blue')

    # get the rotation of the roi based on the tissue boundary
    angle = detect_rotation(filename, anno, tissue_threshold)
    if angle is None:
        print("WARNING: No Slide Background Detected in ROI. Skipping ROI...")
        return None

    # load the roi into memory, rotate it by the calculated angle
    frame_rot, frame_loc, crop_loc = rotate_roi(loader, anno, angle)
    if args.write_roi:
        roi_f = sana_io.get_ofname(filename, args.ofiletype, '_ORIG_%d' % count,
                                   args.odir, args.rdir)
        frame_rot.save(roi_f)

    # generate the tissue mask from the rotated ROI
    min_body_size, min_hole_size = 1e5, 1e5
    detector = TissueDetector(loader.mpp, loader.ds, loader.lvl)
    detector.run(frame_rot.copy(), tissue_threshold,
                 min_body_size, min_hole_size)
    tissue_mask = detector.generate_mask(frame_rot.size)

    # separate the stain from the ROI using color deconvolution
    frame_stain = separate_roi(frame_rot, args.stain, args.target)
    if args.write_stain:
        stain_f = sana_io.get_ofname(filename, args.ofiletype, '_STAIN_%d' % count,
                                     args.odir, args.rdir)
        frame_stain.save(stain_f)

    # threshold the stain separated ROI
    frame_thresh = threshold_roi(frame_stain, tissue_mask, blur=1)
    if args.write_thresh:
        thresh_f = sana_io.get_ofname(filename, args.ofiletype, '_THRESH_%d' % count,
                                      args.odir, args.rdir)
        frame_thresh.save(thresh_f)

    # get the density using a tiled analysis
    # NOTE: previously was using frame_stain NOT frame_thresh
    frame_density, tds = density_roi(loader, frame_thresh,
                                     args.tsize, args.tstep, round=True)
    if args.write_density:
        density_f = sana_io.get_ofname(filename, args.ofiletype, '_DENSITY_%d' % count,
                                       args.odir, args.rdir)
        frame_density.save(density_f)

    # get the layer 0 detections from the tissue mask
    x_0, y_0 = detect_layer_0_roi(tissue_mask)

    # get the layer 6 detections from the stain density
    x_6, y_6 = detect_layer_6_roi(frame_density, gm_threshold, tds)

    # smooth the detections
    y_0 = signal.medfilt(y_0, 71)

    # TODO: upscale and interpolate
    y_6 = signal.medfilt(y_6, 31)

    # build the gm detection by combining layers 0 and 6
    x = np.concatenate([x_0[::-1], x_6])
    y = np.concatenate([y_0[::-1], y_6])
    gm = sana_geo.Polygon(x, y, loader.mpp, loader.ds,
                          is_micron=False, lvl=loader.lvl)

    # revert to original orientation, location, and resolution
    gm.to_float()
    gm.translate(-(crop_loc+frame_loc))
    gm.rescale(0)
    gm = gm.rotate(anno.centroid()[0], -angle)

    # remove any detections outside the given ROI
    gm = gm.filter(anno)

    # connect the last point to the first point
    gm.connect()

    return gm, angle
#
# end of segment_gm

def process_gm(gm_seg):
    pass
#
# end of process_gm

# detects the rotation the frame based on the linear regression of the estimated tissue boundary
def detect_rotation(filename, anno, tissue_threshold):

    # initialize the Loader at thumbnail resolution
    loader = Loader(filename)
    loader.set_lvl(loader.lc-1)

    # rescale annotation to thumbnail resolution
    anno.rescale(loader.lvl)

    # define the frame size and location based on the centroid and radius
    #   of the polygonal annotation
    centroid, radius = anno.centroid()
    fsize = sana_geo.Point(2*radius, 2*radius, loader.mpp, loader.ds,
                           is_micron=False, lvl=loader.lvl)
    flocs = [sana_geo.Point(centroid[0]-radius, centroid[1]-radius,
                            loader.mpp, loader.ds,
                            is_micron=False, lvl=loader.lvl)]

    # initalize the Framer and load the frame into memory
    framer = Framer(loader, fsize, locs=flocs)
    frame = framer.load(0)

    # generate the tissue detections using the pre-calculated threshold
    min_body_size, min_hole_size = 1e5, 1e5
    detector = TissueDetector(loader.mpp, loader.ds, loader.lvl)
    detector.run(frame, tissue_threshold, min_body_size, min_hole_size)

    # shift the annotation to the relative position
    anno.translate(framer.locs[0])
    centroid -= framer.locs[0]

    # get all the tissue detection vertices that are within the annotation
    layer_0 = detector.ray_trace_vertices(anno)
    if layer_0.n == 0:
        return None

    # revert annotation to the original dimension
    anno.translate(-framer.locs[0])
    anno.rescale(0)

    # do linear regression to find the best fit for the pial boundary
    m0, b0 = layer_0.linear_regression()
    a = sana_geo.Point(0, m0*0 + b0, loader.mpp, loader.ds,
                       is_micron=False, lvl=loader.lvl)
    b = sana_geo.Point(frame.size[0], m0*frame.size[0] + b0,
                       loader.mpp, loader.ds,
                       is_micron=False, lvl=loader.lvl)

    # find the rotation of the best fit
    angle = sana_geo.find_angle(a, b)

    # transform the III and IV quadrant to I and II respectively
    quadrant = angle//90
    if quadrant > 1:
        angle -= 180

    # rotate layer 0 detection, modify the angle such that layer 0 is that the
    #  top of the frame instead of the left or right
    layer_0_rot = layer_0.rotate(centroid, angle)
    if np.mean(layer_0_rot.y) >= frame.size[1]//2:
        angle -= 180
    return angle
#
# end of detect_rotation

def rotate_roi(loader, anno, angle):

    # rescale the annotation to the given pixel resolution
    anno.rescale(loader.lvl)

    # define the frame size and location based on the centroid and radius
    #   of the polygonal annotation
    centroid, radius = anno.centroid()
    fsize = sana_geo.Point(2*radius, 2*radius, loader.mpp, loader.ds,
                           is_micron=False, lvl=loader.lvl)
    flocs = [sana_geo.Point(centroid[0]-radius, centroid[1]-radius,
                            loader.mpp, loader.ds,
                            is_micron=False, lvl=loader.lvl)]

    # initalize the Framer and load the frame
    framer = Framer(loader, fsize, locs=flocs)
    frame = framer.load(0)

    # shift the annotation to the relative position
    anno.translate(framer.locs[0])
    centroid -= framer.locs[0]

    # rotate the frame
    frame_rot = frame.rotate(angle)
    anno_rot = anno.rotate(centroid, angle)

    # crop the frame based on the new bounding box of the ROI
    crop_loc, crop_size = anno_rot.bounding_box()
    frame_rot = frame_rot.crop(crop_loc, crop_size)

    # revert the annotation to the original scale
    anno.translate(-framer.locs[0])
    anno.rescale(0)

    return frame_rot, framer.locs[0], crop_loc
#
# end of rotate_roi

def separate_roi(frame, stain_type, stain_target):

    # initialize the color deconvolution algorithm
    separator = StainSeparator(stain_type, stain_target)

    # run it on the given frame
    return Frame(separator.run(frame.img)[0])
#
# end of separate_roi

def threshold_roi(frame, tissue_mask, blur=0):

    # perform the thresholding
    thresholder = CellThresholder(frame, tissue_mask, blur, mx=245)
    thresholder.mask_frame()
    return thresholder.frame
#
# end of threshold_roi

def density_roi(loader, frame, tsize, tstep, round=False):

    # initialize the tiler
    tsize = sana_geo.Point(tsize[0], tsize[1], loader.mpp, loader.ds)
    tstep = sana_geo.Point(tstep[0], tstep[1], loader.mpp, loader.ds)
    tiler = Tiler(loader, tsize, tstep)

    # load the tiles
    tiles = tiler.load_tiles(frame, pad=True)

    # calculate the density of the stain in the roi
    density = Frame(np.mean(tiles, axis=(2, 3)))
    if round:
        density.round()

    return density, tiler.ds

# TODO: might need to shift the detection based on the amount of blur?
def detect_layer_0_roi(tissue_mask):

    # get the infra section of the tissue_mask
    infra = tissue_mask.img[:tissue_mask.img.shape[0]//2, :]

    # loop through the columns of the mask
    x, y = [], []
    for i in range(infra.shape[1]):

        # last index of zeros in the mask is the start of tissue
        inds = np.argwhere(infra[:, i] == 0)
        if len(inds) != 0:
            x.append(i)
            y.append(inds[-1][0])

    return x, y

def detect_layer_6_roi(density, threshold, tds):

    # normalize the density frame by the max of each column
    density.img = density.img / np.max(density.img, axis=0)

    # get the supra layer of the roi
    # TODO: should we normalize by the supra max?
    supra = density.img[density.img.shape[0]//2:, :]

    # loop through the columns in the supra region
    x, y = [], []
    for i in range(supra.shape[1]):

        # first index where supra is below threshold is the end of the GM
        # NOTE: these values are being scaled up from the tile dimension
        inds = np.argwhere(supra[:, i] <= threshold)
        if len(inds) != 0:
            x.append(i * tds[0])
            y.append((density.img.shape[0]//2 + inds[0][0]) * tds[1])

    return x, y

#
# end of file
