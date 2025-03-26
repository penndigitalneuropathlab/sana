
# system packages
import os
import time

# openslide importing
OPENSLIDE_PATH = os.environ.get('OPENSLIDE_DLL_DIRECTORY')
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
    
# installed packages
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# sana packages
import sana.image
import sana.geo
import sana.logging

class FileNotSupported(Exception):
    def __init__(self, f):
        self.message = 'Unable to load file: %s' % f
        super().__init__(self.message)

class FileNotFound(Exception):
    def __init__(self, f):
        self.message = 'File not found: %s' % f
        super().__init__(self.message)

class Loader(openslide.OpenSlide):
    """ Provides a wrapper to OpenSlide which enables users to easily load Frames of slide data
    :param logger: processing parameters are stored here
    :param fname: filepath to the slide file to load
    :param mpp: resolution in microns per pixel, usually is stored in the metadata
    """
    def __init__(self, logger: sana.logging.Logger, fname: str, mpp: float=None):
        self.logger = logger        
        self.fname = fname
        
        if not os.path.exists(self.fname):
            raise FileNotFoundError(self.fname)

        # initialize the slide
        try:
            super().__init__(self.fname)
        except Exception as e:
            print(e)
            raise FileNotSupported(self.fname)

        # define the necessary attributes
        self.ds = np.array(self.level_downsamples)

        if mpp is None:
            if self.fname.endswith('.svs') and 'aperio.MPP' in self.properties:
                self.mpp = float(self.properties['aperio.MPP'])
            elif 'openslide.mpp-x' in self.properties:
                self.mpp = float(self.properties['openslide.mpp-x'])
            else:
                self.mpp = None
        else:
            self.mpp = mpp

        # store the slide bounding box information
        if 'openslide.bounds-x' in self.properties:
            self.bx = float(self.properties['openslide.bounds-x'])
            self.by = float(self.properties['openslide.bounds-y'])            
            self.bw = float(self.properties['openslide.bounds-width'])
            self.bh = float(self.properties['openslide.bounds-height'])
        else:
            self.bx = 0
            self.by = 0            
            self.bw = self.level_dimensions[0][1]
            self.bh = self.level_dimensions[0][0]
        self.bloc = sana.geo.Point(self.bx, self.by, is_micron=False, level=0)
        self.bsize = sana.geo.Point(self.bw, self.bh, is_micron=False, level=0)

        # define the converter object to convert units and rescale data
        self.converter = sana.geo.Converter(self.mpp, self.ds)

    def load_thumbnail(self):
        """
        Loads the thumbnail using the image stored at the top of the pyramid
        """
        self.thumbnail_level = self.level_count - 1
        w, h = self.level_dimensions[self.thumbnail_level]
        loc = sana.geo.Point(0, 0, is_micron=False, level=self.thumbnail_level)
        size = sana.geo.Point(w, h, is_micron=False, level=self.thumbnail_level)
        return self.load_frame(loc, size, level=self.thumbnail_level)

    def load_frame(self, loc: sana.geo.Point, size: sana.geo.Point, level: int=0, pad_color=0):
        """
        Loads a Frame into memory, defined by the top left corner and a requested size
        :param loc: top left corner of Frame (in any unit)
        :param size: size of Frame to load (in any unit)
        :param level: level resolution to use
        :param pad_color: uint8 color to use if padding is required (default=0)
        """
        if type(pad_color) is int:
            pad_color = (pad_color,pad_color,pad_color)
        loc = loc.copy()
        size = size.copy()

        # ensure correct pixel resolution
        loc = self.converter.to_pixels(loc, level)
        size = self.converter.to_pixels(size, level)

        # prepare variables to calculate padding
        w, h = self.level_dimensions[level]
        padx1, pady1, padx2, pady2 = 0, 0, 0, 0

        # special case: requested frame is entirely outside slide coordinates
        if (loc[0] + size[0]) <= 0: # left bound
            padx1 = int(size[0])
            loc[0] = 0
            size[0] = 0
        if (loc[1] + size[1]) <= 0: # upper bound
            pady1 = int(size[1])
            loc[1] = 0
            size[1] = 0
        if (loc[0] - size[0]) >= w: # right bound
            padx2 = int(size[0])
            loc[0] = w
            size[0] = 0
        if (loc[1] - size[1]) >= h: # lower bound
            pady2 = int(size[1])
            loc[1] = h
            size[1] = 0

        # normal case: requested frame is partially outside slide coordinates
        if loc[0] < 0: # left bound
            padx1 = -int(loc[0])
            loc[0] = 0
            size[0] -= padx1
        if loc[1] < 0: # upper bound
            pady1 = -int(loc[1])
            loc[1] = 0
            size[1] -= pady1
        if (loc[0] + size[0]) > w: # right bound
            padx2 = int(loc[0] + size[0] - w)
            size[0] -= padx2
        if (loc[1] + size[1]) > h: # lower bound
            pady2 = int(loc[1] + size[1] - h)
            size[1] -= pady2

        # load the region of interest using OpenSlide
        # NOTE: OpenSlide requires the loc to be in original resolution (i.e. level=0)
        loc = self.converter.rescale(loc, 0)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        im = self.read_region(location=loc, level=level, size=size)
        loc = self.converter.rescale(loc, level)

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
        return sana.image.Frame(img, level=level, converter=self.converter)

    def load_frame_with_roi(self, roi: sana.geo.Polygon, level: int=0, padding=0):
        """
        Loads a Frame into memory based on the bounding box of the input ROI
        :param roi: polygon to be used to load the Frame
        :param level: level resolution to use        
        :param padding: amount of padding to add context to the ROI
        """
        roi = roi.copy()
        self.logger.data["angle"] = None        
        self.logger.data["crop_loc"] = None
        self.logger.data["crop_size"] = None
        self.logger.data["M"] = None
        self.logger.data["nw"] = None
        self.logger.data["nh"] = None
        
        # scale the roi to the proper resolution
        roi = self.converter.rescale(roi, level)

        # get the bounding box of the ROI
        loc, size = roi.bounding_box()

        # shift loc and increase size to add a padded border around the bounding box
        loc -= padding
        size += 2*padding

        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)

        # load the frame into memory
        self.logger.debug('Loading Frame from .svs slide file...')
        t0 = time.time()
        frame = self.load_frame(loc, size, level=level)
        self.logger.debug('Done I/O (%.2f sec)' % (time.time() - t0))

        # set the amount of padding used
        frame.frame_padding = padding

        # store the processing parameters
        self.logger.data['loc'] = loc
        self.logger.data['size'] = size
        self.logger.data['padding'] = padding
        
        if self.logger.debug_level == 'full':
            fig, ax = plt.subplots(1,1)
            ax.imshow(frame.img)
            roi.translate(loc)            
            ax.plot(*roi.T, color='red')
            fig.suptitle(f'loc={loc}, size={size}')

        return frame

    def load_frame_with_segmentations(self, top: sana.geo.Curve, right: sana.geo.Curve, bottom: sana.geo.Curve, left: sana.geo.Curve, level: int=0, padding=0):
        """
        Loads a Frame into memory based on a pseudo-quadrilateral ROI. The vertices in the segments should be in a clockwise order (i.e. ROI is defined by top -> right -> bottom -> left)
        :param top: top segment of the ROI, resulting Frame will be rotated such that this segment is horizontal and at the top of the image
        :param right: right segment of the ROI
        :param bottom: bottom segment of the ROI
        :param left: left segment of the ROI
        :param level: level resolution to use
        :param padding: amount of padding to add context to the ROI

        # TODO: make a before/after image instead!
        ![image](../../sana/examples/images/ex0_SEG.png)
        """
        top = self.converter.rescale(top.copy(), level)
        right = self.converter.rescale(right.copy(), level)
        bottom = self.converter.rescale(bottom.copy(), level)
        left = self.converter.rescale(left.copy(), level)
        
        # rotate to make the top curve horizontal
        origin = sana.geo.point_like(top, 0, 0)
        angle = top.get_angle()
        [curve.rotate(origin, -angle) for curve in [top, right, bottom, left]]
            
        # make sure top is above bottom after rotation
        if np.mean(top[:,1]) > np.mean(bottom[:,1]):
            [curve.rotate(origin, 180) for curve in [top, right, bottom, left]]
            angle += 180
    
        # create a bounding box ROI from the rotated curves
        xmi, ymi = np.min([np.min(curve, axis=0) for curve in [top, right, bottom, left]], axis=0)
        xmx, ymx = np.max([np.max(curve, axis=0) for curve in [top, right, bottom, left]], axis=0)
        loc = sana.geo.point_like(top, xmi, ymi)
        size = sana.geo.point_like(top, xmx-xmi, ymx-ymi)
        roi = sana.geo.rectangle_like(top, loc, size)

        # apply padding to the rotated ROI
        loc -= padding
        size += 2*padding
        roi = sana.geo.rectangle_like(top, loc, size)

        # rotate to the original coordinate space
        roi.rotate(origin, angle)
        [curve.rotate(origin, angle) for curve in [top, right, bottom, left]]

        # load the frame using the ROI
        frame = self.load_frame_with_roi(roi, level=level, padding=0)
        roi.translate(self.logger.data['loc'])
        if self.logger.debug_level == 'full':
            fig, axs = plt.subplots(1,3, figsize=(10,5))
            ax = axs[0]
            ax.imshow(frame.img)
            [curve.translate(self.logger.data['loc']) for curve in [top, right, bottom, left]]
            ax.plot(*top.T, color='red', label='top')
            ax.plot(*right.T, color='magenta', label='right')
            ax.plot(*bottom.T, color='blue', label='bottom')
            ax.plot(*left.T, color='green', label='left')
            ax.set_title('Original')
            ax.legend(loc="center")

        # rotate the frame using the angle we've already determined
        M, nw, nh = frame.get_rotation_matrix(angle)
        frame.warp_affine(M, nw, nh)
        roi.transform(M)
        if self.logger.debug_level == 'full':
            ax = axs[1]
            ax.imshow(frame.img)
            [curve.transform(M) for curve in [top, right, bottom, left]]
            ax.plot(*top.T, color='red', label='top')
            ax.plot(*right.T, color='magenta', label='right')
            ax.plot(*bottom.T, color='blue', label='bottom')
            ax.plot(*left.T, color='green', label='left')
            ax.set_title('Rotated')
            ax.legend(loc="center")

        # crop the frame using the bounding box of the rotated roi
        crop_loc, crop_size = roi.bounding_box()
        frame.crop(crop_loc, crop_size)
        roi.translate(crop_loc)
        if self.logger.debug_level == 'full':
            ax = axs[2]
            ax.imshow(frame.img)
            [curve.translate(crop_loc) for curve in [top, right, bottom, left]]
            ax.plot(*top.T, color='red', label='top')
            ax.plot(*right.T, color='magenta', label='right')
            ax.plot(*bottom.T, color='blue', label='bottom')
            ax.plot(*left.T, color='green', label='left')
            ax.set_title('Cropped')
            ax.set_xlim([0, frame.img.shape[1]])
            ax.set_ylim([frame.img.shape[0], 0])
            ax.legend(loc="center")
            fig.tight_layout()

        # set the amount of padding used
        frame.frame_padding = padding

        # store the processing parameters
        self.logger.data['angle'] = angle
        self.logger.data['M'] = M
        self.logger.data['nw'] = nw
        self.logger.data['nh'] = nh
        self.logger.data['crop_loc'] = crop_loc
        self.logger.data['crop_size'] = crop_size

        return frame

# loads a series of Frames into memory from a slide image
# use locs argument to define a list of locations to load at
class Framer:
    """ Loads a series of Frames into memory from the slide Lnnoader. 
    :param loader: used to access the WSI information
    :param size: specifies size of Frame to load
    :param level: level resolution to use    
    :param step: distance between Frames, defaults to frame size
    :param fpad: amount of padding to add
    :param rois: list of polygons used to create a frame mask
    """
    def __init__(self, loader: Loader, size: sana.geo.Point, level: int=0, step: sana.geo.Point=None, fpad: sana.geo.Point=None, rois: [sana.geo.Polygon]=[]):

        # store the slide loader
        self.loader = loader
        self.converter = self.loader.converter
        self.level = level
        
        # set the frame size
        self.size = size
        self.converter.to_pixels(self.size, self.level)
        self.size = self.converter.to_int(self.size)
        
        # set the frame step
        if step is None:
            self.step = self.size.copy()
        else:
            self.step = step
        self.converter.to_pixels(self.step, self.level)
        self.step = self.converter.to_int(self.step)

        # set the frame pad amount
        if fpad is None:
            fpad = sana.geo.Point(0, 0, False, self.level)
        self.converter.to_pixels(fpad, self.level)
        self.fpad = self.converter.to_int(fpad)

        # calculate the number of frames in the slide
        slide_size = sana.geo.Point(*self.loader.level_dimensions[self.level], is_micron=False, level=self.level)
        self.nframes = np.ceil(slide_size / self.step).astype(int)

        # prepare the ROIs
        self.rois = [self.converter.rescale(roi, self.level) for roi in rois]

    def load(self, i, j):
        x = i * self.step[0]
        y = j * self.step[1]
        loc = sana.geo.point_like(self.size, x, y) - self.fpad
        size = self.size + 2*self.fpad
        frame = self.loader.load_frame(loc, size, level=self.level)
        frame.frame_padding = self.fpad

        # create the mask for the frame
        for roi in self.rois:
            roi.translate(loc)

        mask = sana.image.create_mask_like(frame, self.rois)
        for roi in self.rois:
            roi.translate(-loc)
        
        return frame, mask
