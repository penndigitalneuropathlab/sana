
import numpy as np
from copy import copy
from matplotlib import pyplot as plt

import sana_geo
from sana_loader import Loader
from sana_color_deconvolution import StainSeparator
from sana_thresholder import TissueThresholder, CellThresholder
from sana_detector import TissueDetector
from sana_framer import Framer
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

def detect_layer_0_roi(filename, anno, tissue_threshold):

    # initialize the Loader
    loader = Loader(filename)
    loader.set_lvl(loader.lc-1)

    # rescale the annotation to the current pixel resolution
    anno.rescale(loader.lvl)

    # calculate the centroid and radius of the annotation
    centroid, radius = anno.centroid()

    # initialize the framer for the annotation roi
    fsize = sana_geo.Point(2*radius, 2*radius, loader.mpp, loader.ds,
                           is_micron=False, lvl=loader.lvl)
    flocs = [sana_geo.Point(centroid[0]-radius, centroid[1]-radius,
                            loader.mpp, loader.ds,
                            is_micron=False, lvl=loader.lvl)]
    framer = Framer(loader, fsize, locs=flocs)

    # load the frame
    frame = framer.load(0)

    # generate the tissue detections using the pre-calculated threshold
    frame.to_gray()
    frame.gauss_blur(5)
    frame.threshold(tissue_threshold, x=255, y=0)
    detector = TissueDetector(loader.mpp, loader.ds, loader.lvl)
    min_body_size = 1e5
    detector.run(frame, min_body_size)

    # shift the annotation to the relative position
    anno.translate(framer.locs[0])

    # get all the tissue detection vertices that are within the annotation
    layer_0 = detector.ray_trace_vertices(anno)

    # shift the annotation back to the absolute position
    anno.translate(-framer.locs[0])

    return layer_0, detector.generate_mask(fsize)
#
# end of detect_layer_0_roi

def rotate_roi(loader, anno, layer_0):

    # rescale the annotation to the current pixel resolution
    anno.rescale(loader.lvl)

    # rescale the vertices in the layer_0 detection
    layer_0.rescale(loader.lvl)

    # calculate the centroid and radius of the annotation
    centroid, radius = anno.centroid()

    # initialize the framer for the current annotation
    fsize = sana_geo.Point(2*radius, 2*radius, loader.mpp, loader.ds,
                           is_micron=False, lvl=loader.lvl)
    flocs = [sana_geo.Point(centroid[0]-radius, centroid[1]-radius,
                            loader.mpp, loader.ds,
                            is_micron=False, lvl=loader.lvl)]
    framer = Framer(loader, fsize, locs=flocs)

    # shift the annotation to the relative position
    anno.translate(framer.locs[0])
    centroid, radius = anno.centroid()

    # load the frame
    frame = framer.load(0)

    # do linear regression to find the best fit for the pial boundary
    m0, b0 = layer_0.linear_regression()
    a = sana_geo.Point(0, m0*0 + b0, loader.mpp, loader.ds,
                       is_micron=False, lvl=loader.lvl)
    b = sana_geo.Point(frame.size[1], m0*frame.size[1] + b0,
                       loader.mpp, loader.ds,
                       is_micron=False, lvl=loader.lvl)

    # find the rotation of the best fit
    angle = sana_geo.find_angle(a, b)
    angle = angle - (90 * (angle//90))

    # TODO: calcualte rotation based on quadrant, don't do the line above
    # rotate layer 0 detection, modify the angle such that layer 0 is that the
    #  top of the frame instead of the left or right
    layer_0_rot = layer_0.rotate(centroid, angle)
    # if np.mean(layer_0_rot.x) >= frame.size[0]//2:
    #     angle += 90
    # else:
    #     angle += 270

    # rotate the frame, roi, and layer 0 detection
    frame_rot = frame.rotate(angle)
    anno_rot = anno.rotate(centroid, angle)

    # crop the frame based on the new bounding box of the ROI
    loc, size = anno_rot.bounding_box()
    frame_rot = frame_rot.crop(loc, size)
    anno_rot.translate(loc)
    anno_rot.round()

    return frame, anno, frame_rot, anno_rot, a, b
#
# end of rotate_roi

# TODO: all below functions should probabbly be in Frame
def get_tissue_mask(frame, tissue_threshold, mpp, ds, lvl):
    tissue_frame = frame.copy()
    tissue_frame.to_gray()
    tissue_frame.gauss_blur(5)
    tissue_frame.threshold(tissue_threshold, x=255, y=0)
    detector = TissueDetector(mpp, ds, lvl)
    min_body_size = 1e5
    detector.run(tissue_frame, min_body_size)
    return detector.generate_mask(tissue_frame.size)

def separate_roi(frame, stain_type, stain_target):

    # initialize the color deconvolution algorithm
    separator = StainSeparator(stain_type, stain_target)

    # run it on the given frame
    return Frame(separator.run(frame.img)[0])
#
# end of separate_roi

def threshold_roi(frame, tissue_mask, blur=0):

    # perform the thresholding
    thresholder = CellThresholder(frame, tissue_mask, blur)
    thresholder.mask_frame()
#
# end of threshold_roi

#
# end of file
