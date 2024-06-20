
# system packages
import os
from copy import copy
import time

# installed packages
import openslide
import numpy as np
from matplotlib import pyplot as plt

# sana packages
import sana.image
import sana.geo

class FileNotSupported(Exception):
    def __init__(self, f):
        self.message = 'Unable to load file: %s' % f
        super().__init__(self.message)

class FileNotFound(Exception):
    def __init__(self, f):
        self.message = 'File not found: %s' % f
        super().__init__(self.message)

# TODO: where to add logger
# TODO: where to add params
class Loader(openslide.OpenSlide):
    """ Provides a wrapper to OpenSlide which enables users to easily load Frames of slide data via various methods and inputs.
    :param fname: filepath to the slide file to load
    :param logger: sana.logger.Logger object
    :param get_thumb: whether or not to preload the thumbnail
    :param mpp: microns per pixel constant, usually is loaded via the aperio.MPP parameter
    """
    def __init__(self, fname, logger, get_thumb=True, mpp=None):
        self.fname = fname
        self.logger = logger

        if not os.path.exists(self.fname):
            raise FileNotFoundError(self.fname)
        
        # initialize the object with the slide
        try:
            super().__init__(self.fname)
        except:
            raise FileNotSupported(self.fname)

        # define the necessary attributes
        self.lc = self.level_count
        self.dim = self.level_dimensions
        self.ds = self.level_downsamples

        # set the conversion scaler for microns per pixel
        if not mpp is None:
            self.mpp = mpp
        else:
            try:
                self.mpp = float(self.properties['aperio.MPP'])
            except:
                self.mpp = 1.0

        # define the converter object to convert units and rescale data
        self.converter = sana.geo.Converter(self.mpp, self.ds)
        self.slide_threshold = None
        self.slide_color = (255, 255, 255)

        # preload the thumbnail if necessary
        if get_thumb:
            # pre-load the thumbnail for easy access
            self.thumbnail = self.load_thumbnail()

            # calculate the color of the slide background
            # self.slide_color = copy(self.thumbnail).get_bg_color()

            # calculate the Slide/Tissue threshold
            # TODO: make this much faster!
            # self.csf_threshold = int(get_csf_threshold(copy(self.thumbnail)))
            self.slide_threshold = None

            # self.thumbnail.csf_threshold = self.csf_threshold

        else:
            self.slide_color = None
            self.slide_threshold = None

    def set_level(self, level):
        self.level = level

    def load_thumbnail(self):
        """
        Loads the thumbnail using the image stored at the top of the pyramid
        """
        self.thumbnail_level = self.lc - 1
        w, h = self.dim[self.thumbnail_level]
        loc = sana.geo.Point(0, 0, is_micron=False, level=self.thumbnail_level)
        size = sana.geo.Point(w, h, is_micron=False, level=self.thumbnail_level)
        return self.load_frame(loc, size, self.thumbnail_level)

    def load_frame(self, loc, size, level=None, pad_color=0):
        """
        Loads a Frame into memory, defined by the top left corner and a requested size
        :param loc: top left corner of Frame (in any unit)
        :param size: size of Frame to load (in any unit)
        :param level: resolution to use, overrides the level attribute of the Loader instance
        :param pad_color: uint8 color to use if padding is required (default=0)
        """
        if level is None:
            level = self.level
        if type(pad_color) is int:
            pad_color = (pad_color,pad_color,pad_color)
        loc = copy(loc)
        size = copy(size)

        # make sure loc and size are in correct pixel resolution
        loc = self.converter.to_pixels(loc, level)
        size = self.converter.to_pixels(size, level)

        # prepare variables to calculate padding
        w, h = self.dim[level]
        padx1, pady1, padx2, pady2 = 0, 0, 0, 0

        # special case: requested frame is entirely outside slide coordinates
        if (loc[0] + size[0]) <= 0: # left bound
            padx1 += int(size[0])
            loc[0] = 0
            size[0] = 0
        if (loc[1] + size[1]) <= 0: # upper bound
            pady1 += int(size[1])
            loc[1] = 0
            size[1] = 0
        if (loc[0] - size[0]) >= w: # right bound
            padx2 += int(size[0])
            loc[0] = w
            size[0] = 0
        if (loc[1] - size[1]) >= h: # lower bound
            pady2 += int(size[1])
            loc[1] = h
            size[1] = 0

        # normal case: requested frame is partially outside slide coordinates
        if loc[0] < 0: # left bound
            padx1 -= int(loc[0])
            loc[0] = 0
            size[0] -= padx1
        if loc[1] < 0: # upper bound
            pady1 -= int(loc[1])
            loc[1] = 0
            size[1] -= pady1
        if (loc[0] + size[0]) > w: # right bound
            padx2 += int(loc[0] + size[0] - w)
            size[0] -= padx2
        if (loc[1] + size[1]) > h: # lower bound
            pady2 = int(loc[1] + size[1] - h)
            size[1] -= pady2

        # load the region of interest via OpenSlide
        # NOTE: OpenSlide requires the loc to be in original resolution (i.e. level=0)
        self.converter.rescale(loc, 0)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        im = self.read_region(location=loc, level=level, size=size) 

        # convert to numpy and remove the alpha channel
        img = np.array(im)[:, :, :3]

        # pad img with rows at the top/bottom with cols of img size
        pady1 = np.full_like(img, pad_color, shape=(pady1, img.shape[1], 3))
        pady2 = np.full_like(img, pad_color, shape=(pady2, img.shape[1], 3))
        img = np.concatenate((pady1, img, pady2), axis=0)

        # pad img with cols at the left/right with rows of the padded img size
        padx1 = np.full_like(img, pad_color, shape=(img.shape[0], padx1, 3))
        padx2 = np.full_like(img, pad_color, shape=(img.shape[0], padx2, 3))
        img = np.concatenate((padx1, img, padx2), axis=1)

        # convert to Frame object
        return sana.image.Frame(
            img, level=level, converter=self.converter,
            slide_threshold=self.slide_threshold,
            slide_color=self.slide_color,
        )

    def load_frame_with_parameters(self):
        """
        Loads a Frame into memory using previously logged parameters
        """

        # load the frame using loc and size
        frame = self.load_frame(self.logger.data['loc'], self.logger.data['size'])
        
        # set the padding
        frame.frame_padding = self.logger.data['padding']

        # perform the first rotation
        M, nw, nh = frame.get_rotation_matrix(self.logger.data['angle1'])
        frame.warp_affine(M, nw, nh)

        # perform the cropping
        frame.crop(self.logger.data['crop_loc'], self.logger.data['crop_size'])

        # perform the second rotation
        M, nw, nh = frame.get_rotation_matrix(self.logger.data['angle2'])
        frame.warp_affine(M, nw, nh)

        return frame
    
    def load_frame_with_roi(self, roi, padding=0):
        """
        Loads a Frame into memory based on the bounding box of the input ROI
        :param roi: a geo.Polygon to be used to load the Frame
        :param padding: amount of padding to add context to the ROI
        """
        roi = roi.to_polygon()

        # scale the roi to the current resolution
        self.converter.rescale(roi, self.level)

        # get the bounding box of the ROI
        loc, size = roi.bounding_box()

        # shift loc and increase size to add a padded border around the bounding box
        loc -= padding
        size += 2*padding

        # load the frame into memory
        self.logger.debug('Loading Frame from .svs slide file...')            
        t0 = time.time()
        frame = self.load_frame(loc, size)
        roi.translate(loc)
        self.logger.debug('Done I/O (%.2f sec)' % (time.time() - t0))

        # set the amount of padding used
        frame.frame_padding = padding

        # store the processing parameters
        self.logger.data['slide_threshold'] = self.slide_threshold
        self.logger.data['loc'] = loc
        self.logger.data['size'] = size
        self.logger.data['padding'] = padding

        # debugging plots
        if self.logger.generate_plots:
            fig, axs = plt.subplots(1,1)
            ax = axs
            ax.imshow(frame.img)
            ax.plot(*roi.T, color='red')
            fig.suptitle('result of load_roi_frame() with input roi')

        return frame

    def load_frame_with_segmentations(self, c1, c2, padding=0):
        """
        Loads a Frame into memory using 2 input boundaries. First we get the bounding box of the 2 boundaries, then rotates the image so that c1 is horizontal and at the top of the image
        TODO: add link to example image for docs
        :param c1: geo.Curve that will be rotated to the top of the Frame
        :param c2: geo.Curve that is semi-parallel to c1
        :param padding: amount of padding to apply to the Frame
        """
        c1 = c1.to_curve()
        c2 = c2.to_curve()

        # get the center of the 2 curves
        roi = sana.geo.get_polygon_from_curves(c1, c2)
        center = np.mean(roi, axis=0)

        # get the average of the angles of both curves
        # TODO: or just use c1 or c2?
        c1_angle = c1.get_angle()
        c2_angle = c2.get_angle()
        # TODO: if using average, need to resolve quadrants!!
        angle = (c1_angle + c2_angle) / 2
        angle = c1_angle
        self.logger.debug('Orthogonal Angle Found: (%.2f, %.2f) -- %.2f' % (c1_angle, c2_angle, angle))
    
        # make sure c1 is on top of c2 after rotating
        c1.rotate(center, -angle)
        c2.rotate(center, -angle)
        if np.mean(c1[:,1]) > np.mean(c2[:,1]):
            c1.rotate(center, 180)
            c2.rotate(center, 180)
            angle += 180

        # create a bounding box ROI from the rotated curves
        roi = sana.geo.get_polygon_from_curves(c1, c2)
        loc, size = roi.bounding_box()

        # apply the padding to get the final roi which we will crop to in the rotated coordinate space
        loc -= padding
        size += 2*padding
        roi = sana.geo.rectangle_like(loc, size, c1)

        # rotate the roi back to the original coordinate space
        roi.rotate(center, angle)
        c1.rotate(center, angle)
        c2.rotate(center, angle)

        # load the frame from the ROI
        # NOTE: we've already performed the padding
        frame = self.load_frame_with_roi(roi, padding=0)
        roi.translate(self.logger.data['loc'])
        c1.translate(self.logger.data['loc'])
        c2.translate(self.logger.data['loc'])
        if self.logger.generate_plots:
            fig, axs = plt.subplots(1,3)
            ax = axs[0]
            ax.imshow(frame.img)
            ax.plot(*c1.T, color='red', label='c1')
            ax.plot(*c2.T, color='blue', label='c2')
            ax.set_title('Original Frame loaded from Curves')
            ax.legend()

        # set the amount of padding used
        frame.frame_padding = padding

        # rotate the frame using the angle we've already calculated
        M, nw, nh = frame.get_rotation_matrix(angle)
        frame.warp_affine(M, nw, nh)
        roi.transform(M)
        c1.transform(M)
        c2.transform(M)
        if self.logger.generate_plots:
            ax = axs[1]
            ax.imshow(frame.img)
            ax.plot(*c1.T, color='red', label='c1')
            ax.plot(*c2.T, color='blue', label='c2')
            ax.set_title('Rotated horizontally with c1 on top')
            ax.legend()

        # crop the frame using the bounding box of the rotated roi
        crop_loc, crop_size = roi.bounding_box()
        frame.crop(crop_loc, crop_size)
        roi.translate(crop_loc)
        c1.translate(crop_loc)
        c2.translate(crop_loc)
        if self.logger.generate_plots:
            ax = axs[2]
            ax.imshow(frame.img)
            ax.plot(*c1.T, color='red', label='c1')
            ax.plot(*c2.T, color='blue', label='c2')
            ax.set_title('Final Rotated/Cropped Frame')
            ax.legend()

        # store the processing parameters
        self.logger.data['angle'] = angle
        self.logger.data['M'] = M
        self.logger.data['nw'] = nw
        self.logger.data['nh'] = nh
        self.logger.data['crop_loc'] = crop_loc
        self.logger.data['crop_size'] = crop_size

        return frame
    
    def load_frame_with_landmarks(self, v, w, padding=0):
        """
        Loads a Frame into memory using the extreme points in a landmark vector. First, the extreme points are identified. Then, a rectangle with a given width is found orthogonal to the input vector
        :param v: landmark vector
        :param w: width (in microns) of rectangle to load into memory
        :param padding: amount of padding to apply to the Frame
        """

        # get the angle of the vector based on linear representation
        # NOTE: adding 90 degrees makes it vertical rather than horizontal
        angle = v.get_angle()+90

        # rotate the vector to be vertical
        c = sana.geo.Point(0, 0, False, self.level)
        v.rotate(c, -angle)

        # define the coords of the rotated frame to load in using the extreme y values
        # TODO: w is in microns not pixels!! need to rescale
        xc = np.mean(v[:,0])
        x0 = xc - w//2 - padding//2
        x1 = xc + w//2 + padding//2
        pmin = v[np.argmin(v[:,1])]
        pmax = v[np.argmax(v[:,1])]
        y0 = pmin[1]
        y1 = pmax[1]
        roi = sana.geo.Polygon([x0, x1, x1, x0], [y0, y0, y1, y1], is_micron=False, level=self.level)

        # rotate back to the origin
        v.rotate(c, angle)
        roi.rotate(c, angle)

        # load a frame from the ROI
        # NOTE: we'e already performed the padding
        frame = self.load_frame_with_roi(roi, padding=0)
        v.translate(self.logger.data['loc'])
        roi.translate(self.logger.data['loc'])
        if self.logger.generate_plots:
            fig, axs = plt.subplots(1,3)
            ax = axs[0]
            ax.imshow(frame.img)
            ax.plot(*v.T, 'x', color='red', label='LANDMARKS')
            ax.plot(*roi.T, color='black', label='ROI')
            ax.set_title('Original Frame loaded from Landmarks')
            ax.legend()

        # set the amount of padding used
        frame.frame_padding = padding

        # get the tiles centered on the 2 extreme vertices
        size = sana.geo.Point(50, 50, is_micron=True)
        self.converter.to_pixels(size, self.level)
        t1 = frame.get_tile(pmin-size//2, size)
        t2 = frame.get_tile(pmax-size//2, size)

        # calculate the average pixel brightness
        # NOTE: we just care how close the tile is to white
        t1_intensity = np.mean(t1)
        t2_intensity = np.mean(t2)

        # the tile on the top is darker than the tile on bottom, this means we must rotate 180
        if t1_intensity < t2_intensity:
            angle += 180

        # rotate the image to be vertical
        M, nw, nh = frame.get_rotation_matrix(angle)
        frame.warp_affine(M, nw, nh)
        v.transform(M)
        roi.transform(M)
        if self.logger.generate_plots:
            ax = axs[1]
            ax.imshow(frame.img)
            ax.plot(*v.T, 'x', color='red', label='LANDMARKS')
            ax.plot(*roi.T, color='black', label='ROI')
            ax.set_title('Rotated vertically with slide background on top')
            ax.legend()

        # sort the vector by the y values so that the first landmark is the slide landmark, not tissue
        v = v[np.argsort(v[:,1])]

        # crop the frame using the bounding box of the rotated roi
        crop_loc, crop_size = roi.bounding_box()
        frame.crop(crop_loc, crop_size)
        v.translate(crop_loc)
        roi.translate(crop_loc)
        if self.logger.generate_plots:
            ax = axs[2]
            ax.imshow(frame.img)
            ax.plot(*v.T, 'x', color='red', label='LANDMARKS')
            ax.plot(*roi.T, color='black', label='ROI')
            ax.set_title('Final Rotated/Cropped Frame')
            ax.legend()
        
        # store the processing parameters
        self.logger.data['angle'] = angle
        self.logger.data['M'] = M
        self.logger.data['nw'] = nw
        self.logger.data['nh'] = nh
        self.logger.data['crop_loc'] = crop_loc
        self.logger.data['crop_size'] = crop_size

        return frame, v

    # this function loads a frame of slide data using a given
    # GM Greatest samping zone ROI, which is 4 points 
    def load_gm_zone_frame(self, logger, params, orig_roi, padding=None):

        # create a copy to not disturb the original data
        roi = orig_roi.copy()

        # load the frame from the ROI
        frame = self.load_roi_frame(logger, params, roi, padding, copy=False)

        if logger.plots:
            fig, axs = plt.subplots(1,3)
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            plot_poly(axs[0], roi, color='red')

        # split the ROI into 4 lines and find the line closest to slide background
        angle = get_gm_zone_angle(frame, roi, logger)
        logger.info('Best Orthog. Angle found: %s' % (str(angle)))

        # rotate the image/ROI to orthogonalize them to the CSF boundary
        M1, nw, nh = frame.get_rotation_mat(angle)
        frame.warp_affine(M1, nw, nh)
        roi.transform(M1)

        if logger.plots:
            axs[1].imshow(frame.img)
            plot_poly(axs[1], roi, color='red')

        # crop the frame/ROI to remove borders
        # TODO: this might affect the amount of slide that is found near boundaries
        # TODO: do this at the end?
        orig_size = frame.size()
        crop_loc, crop_size = roi.bounding_box()
        roi.translate(crop_loc)
        frame.crop(crop_loc, crop_size)
        if logger.plots:
            axs[2].imshow(frame.img)
            plot_poly(axs[2], roi, color='red')

        # no extra rotation needed!
        rotate_angle = 0
        M2 = None

        # store the values used during loading the frame
        params.data['slide_threshold'] = self.slide_threshold
        params.data['orig_size'] = orig_size
        params.data['crop_loc'] = crop_loc
        params.data['crop_size'] = crop_size
        params.data['angle1'] = angle
        params.data['angle2'] = rotate_angle
        params.data['M1'] = M1
        params.data['M2'] = M2

        # finally, return the orthogonalized frame
        return frame
    

# loads a series of Frames into memory from a slide image
# use locs argument to define a list of locations to load at
class Framer:
    """ Loads a series of Frames into memory from the slide Loader. 
    :param loader: Loader object
    :param size: (M,N) Point, specifies size of Frame to load
    :param step: (x,y) Point, distance between Frames, defaults to size
    :param fpad: (x,y) Point, amount to pad to the frame
    """
    def __init__(self, loader, size, step=None, fpad=None, rois=[]):

        # store the slide loader
        self.loader = loader
        self.converter = self.loader.converter
        self.level = self.loader.level
        
        # set the frame size
        self.size = size
        self.converter.to_pixels(self.size, self.level)
        self.size = self.converter.to_int(self.size)

        # set the frame step
        if step is None:
            self.step = copy(self.size)
        else:
            self.step = step
        self.converter.to_pixels(self.step, self.lvl)
        self.step = self.converter.to_int(self.step)

        # set the frame pad amount
        if fpad is None:
            fpad = sana.geo.Point(0, 0, False, self.loader.lvl)
        self.converter.to_pixels(fpad, self.lvl)
        self.fpad = self.converter.to_int(fpad)

        # calculate the number of frames in the slide
        slide_size = self.loader.get_dim()
        self.converter.to_pixels(slide_size, self.level)
        slide_size = self.converter.to_int(slide_size)
        self.nframes = np.ceil(slide_size / self.step)

    def load(self, i, j):
        x = i * self.step[0]
        y = j * self.step[1]
        loc = sana.geo.point_like(x, y, self.size) - self.fpad
        size = self.size + 2*self.fpad
        frame = self.loader.load_frame(loc, size)
        frame.frame_padding = self.fpad

        return frame
