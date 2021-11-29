
# system packages
import os
import sys
from copy import copy

# installed packages
import openslide
import numpy as np

# custom packages
import sana_io
from sana_geo import Converter, Point, get_ortho_angle
from sana_frame import Frame, get_csf_threshold, orient_tissue

# provides an interface to initalize and load SVS files
# uses OpenSlide to do this
class Loader(openslide.OpenSlide):
    def __init__(self, fname, calc_thresh=True):

        # initialize the object with the slide
        self.fname = sana_io.get_fullpath(fname)
        super().__init__(self.fname)

        # define the necessary attributes
        self.lc = self.level_count
        self.dim = self.level_dimensions
        self.ds = self.level_downsamples
        self.mpp = float(self.properties['aperio.MPP'])
        self.converter = Converter(self.mpp, self.ds)
        self.csf_threshold = None

        # pre-load the thumbnail for easy access
        thumbnail = self.load_thumbnail()

        # calculate the color of the slide background
        self.slide_color = copy(thumbnail).get_bg_color()

        # calculate the Slide/Tissue threshold
        self.csf_threshold = get_csf_threshold(copy(thumbnail))
    #
    # end of constructor

    # getters
    def get_dim(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        dim = self.dim[lvl]
        return Point(dim[0], dim[1], False, lvl)
    def get_ds(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        return self.ds[lvl]

    # setters
    def set_lvl(self, lvl):
        self.lvl = lvl

    # loads the entire image at the lowest resolution
    def load_thumbnail(self):
        lvl = self.lc - 1
        h, w = self.get_dim(lvl)
        loc = Point(0, 0, False, lvl)
        size = Point(h, w, False, lvl)
        return self.load_frame(loc, size, lvl)

    # loads a Frame into memory, uses a location and size coordinate
    #  image is automatically padded if loc or size exceeds the dimensions
    #  of the SVS slide
    #  -loc: top left corner of Frame (in any unit or resolution)
    #  -size: size of Frame to load (in any unit or resolution)
    #  -lvl: resolution to use, overrides the lvl attribute stored in Loader
    #  -pad_color: color to use to pad the Frame, can use slide_color or white
    def load_frame(self, loc, size, lvl=None, pad_color=None):
        if lvl is None:
            lvl = self.lvl
        if pad_color is None:
            pad_color = (255, 255, 255)

        # make sure loc and size are in current pixel resolution
        self.converter.to_pixels(loc, lvl)
        self.converter.to_pixels(size, lvl)

        # prepare variables to calculate padding
        loc = copy(loc)
        size = copy(size)
        h, w = self.get_dim(lvl)
        padx1, pady1, padx2, pady2 = 0, 0, 0, 0

        # make sure the location isn't negative
        if loc[0] < 0:
            padx1 = int(0 - loc[0])
            loc[0] = 0
            size[0] -= padx1
        if loc[1] < 0:
            pady1 = int(0 - loc[1])
            loc[1] = 0
            size[1] -= pady1

        # make sure the size isn't past the image boundary
        if loc[0] + size[0] > h:
            padx2 = int(loc[0] + size[0] - h)
            size[0] -= padx2
        if loc[1] + size[1] > w:
            pady2 = int(loc[1] + size[1] - w)
            size[1] -= pady2

        # load the region
        # NOTE: upscale the location before accessing the image
        self.converter.rescale(loc, 0)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        img = np.array(self.read_region(
            location=loc, level=lvl, size=size))[:, :, :3]

        # pad img with rows at the top/bottom with cols of img size
        pady1 = np.full_like(img, pad_color, shape=(pady1, img.shape[1], 3))
        pady2 = np.full_like(img, pad_color, shape=(pady2, img.shape[1], 3))
        img = np.concatenate((pady1, img, pady2), axis=0)

        # pad img with cols at the left/right with rows of the padded img size
        padx1 = np.full_like(img, pad_color, shape=(img.shape[0], padx1, 3))
        padx2 = np.full_like(img, pad_color, shape=(img.shape[0], padx2, 3))
        img = np.concatenate((padx1, img, padx2), axis=1)

        return Frame(img, lvl=lvl, converter=self.converter, csf_threshold=self.csf_threshold)
    #
    # end of load_frame

    # this function uses the bounding box of an annotation to load a frame of data
    def load_roi(self, writer, roi):

        # load the frame based on the roi
        loc, size = roi.bounding_box()
        roi.translate(loc)
        frame = self.load_frame(loc, size)

        # store the processing params and return the frame
        writer.data['loc'] = loc
        writer.data['size'] = size
        return frame
    #
    # end of load_roi

    # TODO: implement
    def load_crude_roi(self, writer, roi):
        pass
    #
    # end of load_crude_roi
    
    # this function loads a frame of slide data using a given GM segmentation
    # it uses the boundaries to orthoganilize the frame, then looks for slide
    # background  near the boundaries to orient the tissue boundary to the top of the frame
    # NOTE: this is a beefed version of just using roi.bounding_box() to load the frame
    def load_gm_seg(self, writer, roi):

        # get the angle that best orthogonalizes the segmentation
        angle = get_ortho_angle(roi)

        # TODO: this function has a bug in it, seems to create too big images
        # load the frame based on the segmentation
        loc, size = roi.bounding_centroid()
        roi.translate(loc)
        frame = self.load_frame(loc, size)

        # rotate the segmentation and frame by the ortho angle
        frame.rotate(angle)
        roi.rotate(frame.size()//2, angle)

        # TODO: this might affect the amount of slide that is foound near boundaries
        # crop the frame to remove borders
        crop_loc, crop_size = roi.bounding_box()
        roi.translate(crop_loc)
        frame.crop(crop_loc, crop_size)

        # make sure the tissue boundary is at the top of the image
        angle = orient_tissue(frame, roi, angle)

        # store the processing parameters and return the frame
        writer.data['loc'] = loc
        writer.data['size'] = size
        writer.data['angle'] = angle
        writer.data['crop_loc'] = crop_loc
        writer.data['crop_size'] = crop_size
        return frame
    #
    # end of load_gm_seg
#
# end of Loader

def get_converter(slide):
    slide = openslide.OpenSlide(slide)
    mpp = float(slide.properties['aperio.MPP'])
    ds = slide.level_downsamples
    return Converter(mpp, ds)
#
# end of file
