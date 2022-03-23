
# system packages
import os
import sys
from copy import copy

# installed packages
import openslide
import numpy as np

# custom packages
import sana_io
from sana_geo import Converter, Point, get_ortho_angle, Polygon, plot_poly
from sana_frame import Frame, get_csf_threshold, get_tissue_orientation
from matplotlib import pyplot as plt

class FileNotSupported(Exception):
    def __init__(self):
        self.message = 'File not found or not supported.'
        super().__init__(self.message)

# TODO: make a default loader class which has MPP and ds values
# provides an interface to initalize and load SVS files
# uses OpenSlide to do this
class Loader(openslide.OpenSlide):
    def __init__(self, fname, get_thumb=True):

        # initialize the object with the slide
        self.fname = sana_io.get_fullpath(fname)
        try:
            super().__init__(self.fname)
        except:
            raise FileNotSupported

        # define the necessary attributes
        self.lc = self.level_count
        self.dim = self.level_dimensions
        self.ds = self.level_downsamples
        self.mpp = float(self.properties['aperio.MPP'])
        self.converter = Converter(self.mpp, self.ds)
        self.csf_threshold = None

        if get_thumb:
            # pre-load the thumbnail for easy access
            self.thumbnail = self.load_thumbnail()

            # calculate the color of the slide background
            self.slide_color = copy(self.thumbnail).get_bg_color()

            # calculate the Slide/Tissue threshold
            # TODO: make this much faster!
            self.csf_threshold = get_csf_threshold(copy(self.thumbnail))
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

        # scale the roi to the current resolution
        self.converter.rescale(orig_roi, self.lvl)
        
        # load the frame based on the roi
        loc, size = roi.bounding_box()
        roi.translate(loc)
        frame = self.load_frame(loc, size)

        # store the processing params and return the frame
        writer.data['loc'] = loc
        writer.data['size'] = size
        writer.data['crop_loc'] = Point(0, 0, loc.is_micron, loc.lvl)
        writer.data['crop_size'] = np.copy(size)
        return frame
    #
    # end of load_roi

    # TODO: implement
    def load_crude_roi(self, writer, roi):
        pass
    #
    # end of load_crude_roi

    # TODO: need to pad the segmentation somehow to provide context for tiling
    # this function loads a frame of slide data using a given GM segmentation
    # it uses the boundaries to orthoganilize the frame, then looks for slide
    # background  near the boundaries to orient the tissue boundary to the top of the frame
    # NOTE: this is a beefed version of just using roi.bounding_box() to load the frame
    def load_gm_seg(self, writer, orig_roi):

        # scale the roi to the current resolution
        self.converter.rescale(orig_roi, self.lvl)
        
        roi = copy(orig_roi)
        
        # get the angle that best orthogonalizes the segmentation
        angle = get_ortho_angle(roi)

        # TODO: ensure that everything is done properly here.
        # load the frame based on the segmentation
        loc, size = roi.bounding_box()
        roi.translate(loc)

        orig_frame = self.load_frame(loc, size)
        frame = orig_frame.copy()

        # rotate the image to be vertical
        M1, nw, nh = frame.get_rotation_mat(angle)
        frame.warp_affine(M1, nw, nh)
        roi = roi.transform(M1)

        # TODO: this might affect the amount of slide that is foound near boundaries
        # crop the frame to remove borders
        crop_loc, crop_size = roi.bounding_box()
        roi.translate(crop_loc)
        frame.crop(crop_loc, crop_size)

        # make sure the tissue boundary is at the top of the image
        l1_on_top = get_tissue_orientation(frame, roi, angle)
        if not l1_on_top:
            rotate_angle = 180
        else:
            rotate_angle = 0
        M2, nw, nh = frame.get_rotation_mat(rotate_angle)
        frame.warp_affine(M2, nw, nh)

        # store the processing parameters
        writer.data['loc'] = loc
        writer.data['size'] = size
        writer.data['angle'] = angle
        writer.data['crop_loc'] = crop_loc
        writer.data['crop_size'] = crop_size
        writer.data['M1'] = M1
        writer.data['M2'] = M2

        # finally, return the orthogonalized frame
        return frame
    #
    # end of load_gm_seg

    def from_vector(self, writer, v, l):

        # get the angle of the vector
        angle = v.get_angle()
        angle += 90 # TODO: check why this is necessary

        # rotate the vector to be vertical
        c = Point(0, 0, False, self.lvl)        
        v.rotate(c, angle)
        if not l is None:
            l[0].rotate(c, angle)
            w = np.max(l[0][:,0])-np.min(l[0][:,0])
            l[0].rotate(c, -angle)
        else:
            w = 1600

        # sort the vector by the y values
        v = v[np.argsort(v[:,1])]
        
        # define the coords of the rotated frame to load
        rect = Polygon(np.array([v[0][0]-w//2, v[0][0]+w//2, v[0][0]+w//2, v[0][0]-w//2]),
                       np.array([v[0][1], v[0][1], v[-1][1], v[-1][1]]), False, self.lvl)

        # rotate back to the original orientation
        v.rotate(c, -angle)
        rect.rotate(c, -angle)

        # load a bigger frame based on the rotated rect
        rect_loc, rect_size = rect.bounding_box()
        frame = self.load_frame(rect_loc, rect_size)
        orig_frame = frame.copy()

        # get the local intensities at the vertices
        v.translate(rect_loc)
        x = v[0].astype(int)
        x1, x2, y1, y2 = x[0]-100, x[0]+100, x[1]-100, x[1]+100
        if x1 < 0: x1 = 0
        if x2 > frame.img.shape[1]: x2 = frame.img.shape[1]
        if y1 < 0: y1 = 0
        if y2 > frame.img.shape[0]: y2 = frame.img.shape[0]
        x = frame.img[y1:y2, x1:x2]
        int_first = np.mean(x**2)

        x = v[-1].astype(int)
        x1, x2, y1, y2 = x[0]-100, x[0]+100, x[1]-100, x[1]+100
        if x1 < 0: x1 = 0
        if x2 > frame.img.shape[1]: x2 = frame.img.shape[1]
        if y1 < 0: y1 = 0
        if y2 > frame.img.shape[0]: y2 = frame.img.shape[0]
        x = frame.img[y1:y2, x1:x2]
        int_last = np.mean(x**2)

        v.translate(-rect_loc)
        
        # make sure the intensity of v[0] is higher (this is the glass slide)
        if int_first < int_last:
            angle += 180
            
        # rotate the image to be vertical
        M, nw, nh = frame.get_rotation_mat(angle)
        frame.warp_affine(M, nw, nh)

        # transform the landmarks to fit on the image
        v.translate(rect_loc)
        v = v.transform(M)
        rect.translate(rect_loc)
        rect = rect.transform(M)
        if not l is None:
            for i in range(len(l)):
                l[i].translate(rect_loc)
                l[i] = l[i].transform(M)

        # re-sort the vector by the y values
        v = v[np.argsort(v[:,1])]
                
        # crop the bigger frame by the rect coords
        loc, size = rect.bounding_box()
        x1, x2 = int(loc[0]), int(loc[0]+size[0])
        y1, y2 = int(loc[1]), int(loc[1]+size[1])
        frame.img = frame.img[y1:y2, x1:x2]
        v.translate(loc)
        if not l is None:
            for i in range(len(l)):
                l[i].translate(loc)

        # store the processing parameters and return the frame
        # TODO: change the naming convention
        writer.data['loc'] = rect_loc
        writer.data['size'] = rect_size
        writer.data['angle'] = angle
        writer.data['crop_loc'] = loc
        writer.data['crop_size'] = size
        return frame, v, l, M, orig_frame
    #
    # end of from_vector
#
# end of Loader

def get_converter(slide):
    slide = openslide.OpenSlide(slide)
    mpp = float(slide.properties['aperio.MPP'])
    ds = slide.level_downsamples
    return Converter(mpp, ds)
#
# end of file
