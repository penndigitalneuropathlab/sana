
# system packages
import os
import sys

# installed packages
import cv2
import numpy as np
from PIL import Image, ImageDraw
import nibabel as nib
from matplotlib import colors
from matplotlib import pyplot as plt
import shapely.geometry

# sana packages
import sana.geo
import sana.filter

# TODO: check parameter usage
class Frame:
    """ Provides an interface for applying a series of image processing functions to the stored 2D numpy image array. This class attempts to perform operations in place so that memory usage does not explode.
    :param img: the 2d numpy array to store
    :param level: the slide level the array was loaded from
    :param converter: Converter object used to manipulate units
    :param slide_threshold: grayscale intensity value to remove background slide data
    :param slide_color: color of slide background to use when needed
    :param padding: amount of padding added to the frame
    """
    def __init__(self, img, level=None, converter=None, slide_threshold=None, slide_color=None, padding=0):

        # load/store the image array
        if type(img) is str:
            if img.endswith('.npz'):
                self.img = load_compressed(img)
            else:
                im = np.array(Image.open(img))
                if len(im.shape) == 2:
                    self.img = im[:,:,None]
                else:
                    self.img = im[:,:,:3]
        else:
            self.img = img

        # make sure the image array always has the color channel
        self.check_channels()

        # store the other class attributes
        # TODO: add a docstring for each attribute
        self.level = level
        self.converter = converter
        self.slide_threshold = slide_threshold
        self.slide_color = slide_color
        self.padding = padding

        # TODO:
        self.contours = []

    def check_channels(self):
        """
        Adds a color channel if the image doesn't have one
        """
        if self.img.ndim < 3:
            self.img = self.img[:,:,None]

    def size(self):
        """
        Gets the (w,h) size of the image array
        :returns: Point object
        """
        (h, w) = self.img.shape[:2]
        return sana.geo.Point(w, h, level=self.level)

    def copy(self):
        """
        Generates a copy of the Frame object
        """
        return frame_like(self, np.copy(self.img))

    def is_gray(self):
        """
        Checks if the image has only 1 color channel
        """
        return self.img.shape[2] == 1
    
    def is_rgb(self):
        """
        Checks if the image has 3 color channels
        """
        return self.img.shape[2] == 3

    def is_short(self):
        """
        Checks if the image is in floating point values
        """
        return self.img.dtype == np.uint8

    def is_binary(self):
        """
        Checks if the image has only 0's and 1's in it
        """
        return self.is_gray() and self.is_short() and (np.max(self.img) <= 1) and (np.min(self.img) >= 0)

    def is_float(self):
        return self.img.dtype == float

    def to_float(self):
        """
        Converts short int values to floating point values
        """
        self.img = self.img.astype(float)

    def to_short(self):
        """
        Rounds floating point values to short int values
        """
        self.img = np.rint(self.img).astype(np.uint8)

    def rescale(self, mi=None, mx=None):
        """
        Performs minmax normalization on the image array
        """
        if mi is None:
            mi = np.min(self.img)
        if mx is None:
            mx = np.max(self.img)
        if self.is_short():
            self.to_float()

        self.img = self.img.clip(mi, mx)
        self.img = 255 * (self.img - mi) / (mx - mi)
        self.to_short()

    def to_gray(self, magnitudes=[0.2989, 0.5870, 0.1140]):
        """
        Converts a 3 color image into a 1 color image
        :param magnitudes: color magnitudes to use in the calculation, default is for RGB images
        """
        if self.is_gray():
            return
        if self.is_short():
            self.to_float()
        self.img = np.dot(self.img, magnitudes)
        self.check_channels()

    def get_histogram(self):
        """
        Generates a 256 bin histogram for each color channel
        """
        histogram = np.zeros((256, self.img.shape[-1]))
        for i in range(histogram.shape[-1]):
            histogram[:,i] = np.histogram(self.img[:,:,i], bins=256, range=(0,256))[0]
        return histogram

    # TODO: what is pad doing here
    def get_tile(self, loc, size, pad=False):
        """
        Gets a cropped rectangular tile from the frame
        :param loc: top left of rectangle
        :param size: size of rectangle
        """
        if not self.converter is None:
            self.converter.to_pixels(loc, self.level)
            self.converter.to_pixels(size, self.level)
            loc = self.converter.to_int(loc)
            size = self.converter.to_int(size)
        w, h = self.size()
        (x0, y0) = loc[0], loc[1]
        (x1, y1) = x0+size[0], y0+size[1]

        pad_x0, pad_y0, pad_x1, pad_y1 = 0, 0, 0, 0
        if x0 < 0:
            pad_x0 = -x0
        if y0 < 0:
            pad_y0 = -y0
        if x1 >= w:
            pad_x1 = w-x1
        if y1 >= h:
            pad_y1 = h-y1
        
        x0 = np.clip(x0, 0, None)
        y0 = np.clip(y0, 0, None)
        x1 = np.clip(x1, None, w)
        y1 = np.clip(y1, None, h)
            
        img = self.img[y0:y1, x0:x1, :]
        if pad:
            img = np.pad(img, [
                [pad_y0, pad_y1],
                [pad_x0, pad_x1],
                [0,0],
            ])
        return img

            

    def set_tile(self, loc, size, tile):
        self.converter.to_pixels(loc, self.level)
        self.converter.to_pixels(size, self.level)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        w, h = self.size()
        (x0, y0) = loc[0], loc[1]
        (x1, y1) = x0+size[0], y0+size[1]
        x0 = np.clip(x0, 0, None)
        y0 = np.clip(y0, 0, None)
        x1 = np.clip(x1, None, w)
        y1 = np.clip(y1, None, h)

        self.img[y0:y1, x0:x1, :] = tile

    # TODO: refactor these functions
    def or_tile(self, loc, size, tile):
        self.converter.to_pixels(loc, self.level)
        self.converter.to_pixels(size, self.level)
        loc = self.converter.to_int(loc)
        size = self.converter.to_int(size)
        w, h = self.size()
        (x0, y0) = loc[0], loc[1]
        (x1, y1) = x0+size[0], y0+size[1]
        x0 = np.clip(x0, 0, None)
        y0 = np.clip(y0, 0, None)
        x1 = np.clip(x1, None, w)
        y1 = np.clip(y1, None, h)

        self.img[y0:y1, x0:x1, :] |= tile
        
    def crop(self, loc, size):
        """
        Crops the image to the given rectangle
        """
        self.img = self.get_tile(loc, size)

    def resize(self, size, interpolation=cv2.INTER_NEAREST):
        """
        Resizes the image array to the specified size
        """
        size = self.converter.to_pixels(size, self.level)
        size = self.converter.to_int(size)
        w, h = size
        self.img = cv2.resize(self.img, dsize=(w, h), interpolation=interpolation)
        self.check_channels()

    def apply_morphology_filter(self, filter):
        if not self.is_binary():
            raise ImageTypeException("Cannot apply morphology filter to non-binary image")
        self.img = filter.apply(self.img)[:,:,None]

    def convolve(self, kernel, tile_step=None, do_pad=False, normalize=False):
        """
        Performs a (downsampled) convolution of the given kernel over the image. This is done by creating tile views into the frame, then applying the kernel to each of the views
        """
        if not self.is_gray():
            raise ImageTypeException('Convolution only supported on single-channel images')
        
        if tile_step is None:
            tile_step = sana.geo.Point(1, 1, is_micron=False, level=self.level)

        tile_size = sana.geo.Point(kernel.shape[1], kernel.shape[0], is_micron=False, level=self.level)

        # generate the tile views to convolve the kernel over
        tiles = self.to_tiles(tile_size, tile_step, do_pad=do_pad)

        # perform the convolution
        result = np.sum(tiles * kernel[None,None,:,:], axis=(2,3))

        if normalize:
            ones = frame_like(self, np.ones_like(self.img))
            norm_factor = ones.convolve(kernel, tile_step=tile_step, do_pad=do_pad, normalize=False)
            result /= norm_factor

        return result

    def remove_background(self, min_background=0, max_background=255, debug=False):
        """
        Performs a background subtraction process on a grayscale image. The process works by creating a background image, which is then subtracted by the original image. The background is found by looking at a specified range of pixel values along with convolving a gaussian kernel over the image. Note that background in this case usually refers to non-specific staining data as opposed to glass slide background. It is trivial to remove slide background with a simple threshold, but non-specific staining background can vary throughout the image so some type of adaptive thresholding is necessary. This functions acts somewhat like an adaptive threshold, since we subtract a variable background value throughout the image.

        TODO: add math formula?
        TODO: add example image

        :param min_background: minimum value to be used as background, values below this are often white slide pixels which may throw off the calculation
        :param max_background: maximum value to be used as background, this can be used to exclude the darkest objects in the image so that they don't contribute to background
        :param debug: when True, generates a plot that shows the original image, background image, and processed image
        """
        if self.is_rgb():
            raise ChannelException('Cannot apply background removal in RGB images')

        if debug:
            fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
            axs[0].imshow(self.img)

        # run this function at a lower resolution to make it run faster
        # TODO: if not available, what do we do?
        level = 1
        ds = self.converter.ds[level] / self.converter.ds[self.level]
        ds_size = self.size() / ds

        # run the downsampling
        frame_downsampled = self.copy()
        frame_downsampled.resize(ds_size, interpolation=cv2.INTER_LINEAR)
        frame_downsampled.level = level

        # get the tiles for the convolution
        # TODO: check the parameters, make these arg
        tsize = self.converter.to_int(self.converter.to_pixels(
            sana.geo.Point(100, 100, is_micron=True), 
            self.level
        ))
        if tsize[0] % 2 == 0: tsize[0] += 1
        if tsize[1] % 2 == 0: tsize[1] += 1
        tstep = sana.geo.Point(10, 10, is_micron=True)
        kern_sigma = tsize[0]/5
        kernel = sana.filter.get_gaussian_kernel(tsize[0], kern_sigma)
        fig, ax = plt.subplots(1,1)
        ax.imshow(kernel)

        # find the background simply by gaussian blurring the image
        # NOTE: used to support masking by a minimum+maximum background value but this is not currently used
        background_image = frame_downsampled.convolve(kernel, tstep, do_pad=True, normalize=True)

        # resize the background frame back to original resolution
        background_frame = frame_like(self, background_image)
        background_frame.resize(self.size(), interpolation=cv2.INTER_CUBIC)

        # finally, subtract the background
        # NOTE: we clip to 0 here, this is okay since this will be background data which we don't care about
        self.to_float()
        self.img = self.img - background_frame.img
        np.clip(self.img, 0, 255, out=self.img)
        self.to_short()

        if debug:
            axs[1].imshow(background_frame.img)
            axs[2].imshow(self.img)
            plt.show()

    def rotate(self, angle, interpolation=cv2.INTER_LINEAR):
        """
        Rotates the image TODO around it's center
        :param angle: rotation angle in degrees
        """
        # calculate the rotation matrix, along with the new width and height of the image
        M, w, h = self.get_rotation_matrix(angle)

        # perform the affine transformation
        self.warp_affine(M, w, h, interpolation=interpolation)

    def get_rotation_matrix(self, angle):
        """
        Finds the rotation matrix for the given angle and image center, also calculates the new width and height of the rotated image
        :param angle: rotation angle in degrees
        """

        # get the rotation point
        w, h = self.size()
        cx, cy = w//2, h//2

        # find the rotation matrix
        M = cv2.getRotationMatrix2D(center=(int(cx), int(cy)), angle=angle, scale=1.0)

        # compute the new dimensions
        cos_term = np.abs(M[0,0])
        sin_term = np.abs(M[0,1])
        new_w = int((h * sin_term) + (w * cos_term))
        new_h = int((h * cos_term) + (w * sin_term))

        # translate the rotation matrix so that the image center doesn't move
        M[0,2] += (new_w / 2) - cx
        M[1,2] += (new_h / 2) - cy

        return M, new_w, new_h
    
    def warp_affine(self, M, w, h, interpolation=cv2.INTER_LINEAR):
        """
        Performs an affine transformation using the transformation matrix and the dimensions
        :param M: tranformation matrix
        :param w: width of the image result
        :param h: height of the image result
        """
        if self.is_gray():
            border_value = 0
        else:
            border_value = (255,255,255)
        self.img = cv2.warpAffine(self.img, M,
                                  dsize=(w, h),
                                  flags=interpolation,
                                  borderValue=border_value)
        self.check_channels() # TODO

    def pad(self, pad, alignment='center', mode='symmetric'):
        """
        Pads the image using the specified padding alignment
        :param pad: Point, number of pixels to pad
        :param alignment: 'before', 'center', or 'after' alignment methods
        :param mode: method to use to select padded values, see np.pad
        """
        if alignment == 'before':
            before = pad
            after = sana.geo.point_like(pad, (0,0))
        elif alignment == 'center':
            before = pad//2
            after = pad - before
        elif alignment == 'after':
            before = sana.geo.point_like(pad, (0,0))
            after = pad
        img = np.pad(self.img, ((before[1], after[1]), # y
                                     (before[0], after[0]), # x
                                     (0, 0),                # c
                                     ), mode=mode)
        return frame_like(self, img)
        
    def save(self, fpath, invert_sform=False, spacing=None):
        """
        Writes the image array to a file
        :param fname: filename to write to
        :param invert_sform: see sana.image.save_nifti
        :param spacing: see sana.image.save_nifti
        """
        if fpath.endswith('.nii.gz'):
            self.save_nifti(fpath, invert_sform, spacing)
        else:
            # can't write png's etc as floats, just store the image array
            if self.is_float():
                np.save(os.path.splitext(fpath)[0]+'.npy', self.img)
            else:
                im = self.img
                if self.is_gray():
                    im = self.img[:,:,0]
                if self.is_binary():
                    im *= 255
                im = Image.fromarray(im)
                im.save(fpath)
    
    def save_compressed(self, fpath):
        """
        Creates and saves a compressed binary image array
        """
        if (not self.is_short()) or (not self.is_gray()):
            raise ImageTypeException(f'Array must be short and grayscale -- datatype={self.img.dtype}, shape={self.img.shape}')
        
        if self.is_binary():

            # get the image as a boolean array
            img = self.img[:,:,0].astype(bool)
                
            # pack the bools into bytes (compresses by 1/8)
            arr = np.packbits(img, axis=-1, bitorder='little')
        else:
            arr = self.img[:,:,0]

        # finally, use numpy's compression algorithm to further compress
        np.savez_compressed(fpath, arr)

    def save_nifti(self, fpath, invert_sform=False, spacing=None):
        """
        Writes the image array to a nifti form that programs like ITK-SNAP can read
        :param fname: filename to write to
        :param invert_sform: changes the sign of the sform
        :param spacing: pixel spacing, uses Converter info if not given
        """
        if self.is_float:
            raise DatatypeException('Cannot save floating point image to NifTI')
        
        # get the pixel spacing using the resolution and MPP
        if spacing is None:
            spacing = [self.converter.ds[self.level] * (self.converter.mpp / 1000)] * 2

        # manipulate the channels of the image
        pix = np.expand_dims(pix, (2))
        if self.is_rgb():
            rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            pix = pix.copy().view(dtype=rgb_dtype).reshape(pix.shape[0:3])
        else:
            pix = pix[:,:,:,0]

        nii = nib.Nifti1Image(pix, np.diag([spacing[0], spacing[1], 1.0, 1.0]))
        if invert_sform:
            nii.set_sform([
                [-spacing[0], 0, 0, 0],
                [0, -spacing[1], 0, 0],
                [0, 0, 1, 0],
            ])
        nib.save(nii, fpath)
  
    def mask(self, mask_frame, mask_value=0):
        """
        Applies a binary mask to the image
        :param mask_frame: a binary image frame with the same dimensions
        :param mask_value: value to set the masked data to
        """
        size = self.size()
        mask_size = mask_frame.size()
        if size[0] != mask_size[0] or size[1] != mask_size[1]:
            raise DimensionException('Mask image {} has different dimensions than the frame {}'.format(mask_frame.size(), self.size()))
        
        if self.is_rgb() and type(mask_value) is int:
            mask_value = (mask_value, mask_value, mask_value)
        self.img[mask_frame.img[:,:,0] == 0] = mask_value

    def to_polygons(self):
        if not self.is_binary():
            raise ImageTypeException('Image must be a binary image')

        # run contour detection to find boundaries between objects
        c, h = cv2.findContours(self.img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # TODO: this doesn't split if exactly 1 touching diagonal connection
        # TODO: looks like there are some objects missing?
        objs = []

        # loop through the body contours
        for i in [i for i in range(len(c)) if h[0][i][3] == -1]:
            if c[i].shape[0] < 3:
                objs.append(sana.geo.Polygon(c[i][:,0,0], c[i][:,0,1]))
            else:

                # create the body polygon
                orig_body = shapely.geometry.Polygon(np.squeeze(c[i]))
                new_body = shapely.geometry.Polygon(np.squeeze(c[i]))
                
                # loop through the hole contours in this body contour
                for j in [j for j in range(len(c)) if h[0][j][3] == i]:

                    # create the hole polygon
                    hole = shapely.geometry.Polygon(np.squeeze(c[j]))

                    # remove the hole from the body
                    new_body = new_body.difference(hole)

                # store the resulting body polygons
                orig_res = sana.geo.from_shapely(orig_body, is_micron=False, level=self.level)
                new_res = sana.geo.from_shapely(new_body, is_micron=False, level=self.level)
                if type(new_res) is list:
                    objs += [x for x in new_res if not x is None]
                elif not new_res is None:
                    objs.append(new_res)
                else:
                    objs.append(orig_res)
                                
        return objs

    def instance_segment(self, ctrs, debug=False):
        
        # generate a foreground image using the soma centers
        fg = np.zeros_like(self.img)[:,:,0]
        fg[ctrs[:,1], ctrs[:,0]] = 1

        # create an RGB image for the watershed algorithm, if necessary
        if self.is_gray():
            rgb = np.tile(self.img, 3)
        else:
            rgb = self.img

        # get the unique components
        _, markers = cv2.connectedComponents(fg)

        # run the watershed algorithm on the unique components
        markers = cv2.watershed(rgb, markers)
            
        # remove the background segmentation
        markers += 1
        bg = np.argmax(np.bincount(markers.flatten()))
        markers[markers == bg] = 0

        # convert to markers binary image, preserving boundaries between objects
        binary_markers = markers.copy()
        binary_markers[binary_markers != 0] = 1
        binary_markers = binary_markers.astype(np.uint8)
        binary_markers = frame_like(self, binary_markers)

        # get the object polygons from the instance segmented binary mask
        objs = binary_markers.to_polygons()

        return objs
    
    # NOTE: this function works okay, but heavily relies on the center detection being accurate. For most cells normal segmentation should be fine. For combined cells, we should have an option instead of watershed, segment the combined cells together using ellipses or something like that
    def fit_ellipses(self, ctrs):

        # minimum and maximum radii to search
        min_r = 4
        max_r = 10

        ellipses = []
        for ctr in ctrs:

            # get the local tile of the object
            loc = sana.geo.Point(ctr[0]-max_r, ctr[1]-max_r, is_micron=False, level=self.level)
            size = sana.geo.Point(2*max_r+1, 2*max_r+1, is_micron=False, level=self.level)
            tile = self.get_tile(loc, size)[:,:,0]
            (cx, cy) = (size // 2).astype(int)

            # find the minor axis by inflating a circle until it reaches an edge
            seg = np.zeros_like(tile)
            for minor_r in range(min_r, max_r+1):
                # create the circle segmentation
                circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*minor_r+1, 2*minor_r+1))

                # place the circle in the tile space
                seg[:,:] = 0
                seg[cy-circle.shape[0]//2:cy+circle.shape[0]//2+1, cx-circle.shape[1]//2:cx+circle.shape[1]//2+1] = circle

                # stop growing once the circle finds negative pixels
                if np.sum((seg == 1) & (tile == 1)) < np.sum(seg):
                    break

            # initial best segmentation is just a circle
            best = np.sum(seg == tile)
            best_seg = seg.copy()
            best_angle = 0
            best_major = minor_r

            # generate new segmentations using rotations and major axis values
            n_directions = 12
            for angle in np.linspace(0, 180, n_directions):
                for major_r in range(minor_r+1, max_r+1):

                    # create the rotated ellipse
                    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*minor_r+1, 2*major_r+1))
                    (h, w) = ellipse.shape
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    ellipse = cv2.warpAffine(ellipse, M, (w, h))

                    # place the ellipse in the tile space
                    seg[:,:] = 0
                    seg[cy-ellipse.shape[0]//2:cy+ellipse.shape[0]//2+1, cx-ellipse.shape[1]//2:cx+ellipse.shape[1]//2+1] = ellipse

                    # store the best scoring segmentation
                    score = np.sum(seg == tile)
                    if score > best:
                        best = score
                        best_seg = seg.copy()
                        best_angle = angle
                        best_major = major_r

            # convert the segmentation to a polygon
            c = cv2.findContours(best_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0][:,0,:]
            p = sana.geo.Polygon(*c.T, is_micron=False, level=self.level).connect()
            p.translate(-loc)

            ellipses.append(p)

        return ellipses
    
    # TODO: check parameters
    def anisodiff(self, niter=12, kappa=11, gamma=0.9, step=(5.,5.)):
        """
        Performs anisotrophic diffusion filtering on the image
        https://github.com/agilescientific/bruges/blob/main/bruges/filters/anisodiff.py
        :param: niter: number of iterations
        :param kappa: conduction coefficient
        :param gamma:
        :param step: tuple, distance between pixels
        """
        if self.is_rgb():
            raise ChannelException('Cannot apply filter to RGB image')
        
        self.to_float()

        deltaS = np.zeros_like(self.img)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(self.img)
        gE = gS.copy()

        for ii in range(niter):

            # calculate the diffs (change in pixels in both rows and cols)
            deltaS[:-1, :] = np.diff(self.img, axis=0)
            deltaE[:, :-1] = np.diff(self.img, axis=1)

            # conduction gradients
            gS = 1. / (1. + (deltaS/kappa)**2.) / step[0]
            gE = 1. / (1. + (deltaE/kappa)**2.) / step[1]

            # update matrices
            S = gS * deltaS
            E = gE * deltaE

            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't as questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:, :] -= S[:-1, :]
            EW[:, 1:] -= E[:, :-1]

            self.img += gamma * (NS + EW)

        # clip values between 0 and 255, round to integer
        np.clip(self.img, 0, 255, out=self.img)
        self.to_short()    
    
    def threshold(self, threshold, x, y):
        """
        Applies a threshold to a grayscale image, setting passing pixels to value y and all other pixels to value x
        :param threshold: grayscale value
        :param x: value for below threshold
        :param y: value for above threshold
        """
        if self.is_rgb():
            raise ChannelException('Cannot apply threshold to RGB image')
        
        # preapre image arrays for the true and false conditions
        if type(x) is int:
            x = np.full_like(self.img, x)
        if type(y) is int:
            y = np.full_like(self.img, y)

        # perform the thresholding
        self.img = np.where(self.img < threshold, x, y)

    def closing_filter(self, radius):
        """
        Applies a morphological closing filter. See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        :param radius: radius of the closing kernel
        """
        if not self.is_binary():
            raise ImageTypeException('Morphology filters can only be applied to binary images')
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kern)[:,:,None]

    def opening_filter(self, radius):
        """
        Applies a morphological opening filter. See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        :param radius: radius of the opening kernel
        """
        if not self.is_binary():
            raise ImageTypeException('Morphology filters can only be applied to binary images')
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kern)[:,:,None]

    def get_contours(self, min_body_area=0, min_hole_area=0, max_body_area=np.inf, max_hole_area=np.inf):
        """
        Finds the contours on all edges of the image, then filters the contours based on size criteria. In the image, background is zero pixels and foreground is non-zero pixels. 
        :param min_body_area: minimum size required for a contour to be considered a body
        :param min_hole_area: minimum size required for a contour to be considered a hole
        :param max_body_area: maximum size for a body contour
        :param max_hole_area: maximum size for a hole contour
        :returns: a list of the body contours and a list of the hole contours 
        """
        if not self.is_binary():
            raise ImageTypeException('Contour detection requires a binary image')
        
        self.contours = []
        contours, hierarchies = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchies is None:
            return [], []
        for c, h in zip(contours, hierarchies[0]):
            self.contours.append(Contour(c, h, self.level, self.converter))

        # find the body contours
        for c in self.contours:

            # make sure this contour does not have a parent contour
            if c.hierarchy[3] != -1:
                continue

            # make sure the size matches the criteria
            area = c.polygon.get_area()
            if area >= min_body_area and area <= max_body_area:
                c.body = True

        # now find the hole contours
        for c in self.contours:

            # skip the already found body contours
            if c.body:
                continue

            # make sure this contour has a body contour as it's parent
            if not self.contours[c.hierarchy[3]].body:
                continue

            # make sure the size matches the criteria
            area = c.polygon.get_area()
            if area > min_body_area and area < max_body_area:
                c.hole = True

        bodies = [c for c in self.contours if c.body]
        holes = [c for c in self.contours if c.hole]
        return bodies, holes

    def to_tiles(self, tsize, tstep, do_pad=False):
        if not self.is_gray():
            raise ChannelException(f"Must be a 1 channel image to tile -- dtype={self.img.dtype}")
        
        # convert to correct units
        if not self.converter is None:
            self.converter.to_pixels(tsize, self.level)
            self.converter.to_pixels(tstep, self.level)
            tsize = self.converter.to_int(tsize)
            tstep = self.converter.to_int(tstep)

        # make sure the tile size is odd
        if tsize[0] % 2 == 0: tsize[0] += 1
        if tsize[1] % 2 == 0: tsize[1] += 1

        # pad the frame so that we have center aligned tiles
        if do_pad:
            padsize = tsize - 1
            frame = self.pad(padsize, alignment='center', mode='constant')
        else:
            frame = self

        # calculate the stride lengths
        w, h = frame.size()
        strides = (
            frame.img.itemsize * w * tstep[1], # num bytes between tiles in y direction
            frame.img.itemsize * tstep[0], # num bytes between tiles in x direction
            frame.img.itemsize * w, # num bytes between elements in y direction
            frame.img.itemsize * 1, # num bytes between elements in x direction
        )

        # calculate the output shape
        shape = (
            int(round((h-tsize[1]) // tstep[1])) + 1,
            int(round((w-tsize[0]) // tstep[0])) + 1,
            tsize[1],
            tsize[0],
        )

        # perform the tiling
        # NOTE: setting writeable=False is advised by numpy docs
        return np.lib.stride_tricks.as_strided(frame.img, shape=shape, strides=strides, writeable=False)

class Contour:
    """
    Stores the contour using it's place in the hierarchy and it's polygon
    :param contour: contour array from findContours
    :param hierarchy: hierarchy information from findContours
    :param level: pixel resolution level
    :param converter: Converter object used to manipulate units
    """
    def __init__(self, contour, hierarchy, level, converter):

        # convert the contour array to a Polygon
        self.level = level
        self.polygon = self.contour_to_polygon(contour)
        self.hierarchy = hierarchy
        self.converter = converter

        # flags denoting whether or not this contour is a body or a hole
        self.body = False
        self.hole = False

    def contour_to_polygon(self, contour):
        x = np.array([float(v[0][0]) for v in contour])
        y = np.array([float(v[0][1]) for v in contour])
        polygon = sana.geo.Polygon(x, y, False, self.level)
        return polygon
#
# end of Detection

def frame_like(frame, img):
    """
    Creates a Frame object with the same parameters as the given frame
    :param frame: frame to emulate
    :param img: image array to store in the new Frame object
    """
    return Frame(img,
                 level=frame.level,
                 converter=frame.converter,
                 slide_threshold=frame.slide_threshold,
                 slide_color=frame.slide_color,
                 padding=frame.padding)
#
# end of frame_like

def create_mask_cv2(polygons, frame):
    """
    similar to create_mask(), but uses opencv since PIL was causing some weird issues
    """

    int_polygons = []
    for p in polygons:
        frame.converter.to_pixels(p, frame.level)
        p = frame.converter.to_int(p).connect()
        int_polygons.append(p)

    mask_img = np.zeros(frame.img.shape[:2])
    mask_img = cv2.fillPoly(mask_img, pts=int_polygons, color=1)

    return frame_like(frame, mask_img.astype(np.uint8))

# TODO: PIL is acting weird, merge with cv2 and only use PIL for outline??
def create_mask(polygons, frame, x=0, y=1, holes=[], outlines_only=False, linewidth=1):
    """
    Generates a binary mask from a list of Polygons
    :param polygons: list of body Polygon
    :param frame: Frame object to get the size, level, and converter
    :param x: negative condition value
    :param y: positive condition value
    :param holes: list of hole Polygons
    :param outlines_only: the generated mask will only be the outlines
    :param linewidth: used when outlines_only=True
    """

    # convert the Polygons to lists of values that PIL can handle
    polys = []
    for p in polygons:
        p = p.connect()
        if not frame.converter is None:
            frame.converter.to_pixels(p, frame.level)
            p = frame.converter.to_int(p)
        polys.append([tuple(p[i]) for i in range(p.shape[0])])

    hs = []
    for h in holes:
        if not frame.converter is None:
            frame.converter.to_pixels(h, frame.level)
            h = frame.converter.to_int(h)
        hs.append([tuple(h[i]) for i in range(h.shape[0])])
    
    # create a blank image
    if not frame.converter is None:
        size = frame.converter.to_int(frame.size())
    else:
        size = frame.size()
    mask = Image.new('L', (size[0], size[1]), x)
    dr = ImageDraw.Draw(mask)  

    # draw the body polygons on the image
    for poly in polys:
        if outlines_only:
            if linewidth == 1:
                dr.polygon(poly, outline=y, fill=0)
            else:
                dr.line(poly, fill=y, width=linewidth, joint='curve')
        else:
            #dr.polygon(poly, outline=y, fill=y)
            dr.polygon(poly, fill=y)            

    # draw the hole polygons on the image with the body polygons
    for h in hs:
        dr.polygon(h, outline=x, fill=x)

    # create the Frame object
    return frame_like(frame, np.array(mask)[:, :, None])
#
# end of create_mask

# TODO: this should be in plots.py
def overlay_mask(frame, mask, alpha=0.5, color='red', main_roi_mask=None, sub_roi_masks=[], main_roi=None, sub_rois=[], linewidth=9):
    """
    Overlays a semi-transparent mask on a frame
    :param frame: background image
    :param mask: binary image mask to overlay
    :param alpha: transparency
    :param color: color to use for the mask
    :param main_roi_mask: used to exclude non-annotated pixels
    :param sub_roi_masks: list of masks used to exclude non-annotate pixels
    :param main_roi: if given, overlays the Annotation
    :param sub_rois: if given, overlays the Annotations
    :param linewidth: used when plotting the Annotations
    """
    # get the RGB color
    if type(color) is str:
        color = np.rint(255*np.array(colors.to_rgb(color))).astype(np.uint8)
    overlay = frame.copy()

    if not mask is None:
        mask = mask.copy()
        mask.img = mask.img[:frame.img.shape[0], :frame.img.shape[1]]
    
        # exclude non-annotated pixels
        if not main_roi_mask is None:
            mask.img[main_roi_mask.img == 0] = 0
        if len(sub_roi_masks) != 0:
            full_sub_roi_mask = frame_like(np.zeros_like(mask.img), mask)
            for sub_roi_mask in sub_roi_masks:
                full_sub_roi_mask += sub_roi_mask.img
            mask.img[full_sub_roi_mask == 0] = 0

        # create the overlay image using a weight sum
        overlay.img[mask.img[:,:,0] != 0] = color
        overlay.img = cv2.addWeighted(overlay.img, alpha, frame.img, 1-alpha, 0.0)

    # plot the annotations
    if not main_roi is None:
        main_roi_overlay = create_mask([main_roi], frame, outlines_only=True, linewidth=linewidth)
        overlay.img[main_roi_overlay.img[:,:,0] != 0] = color
    if len(sub_rois) != 0:
        sub_roi_overlay = create_mask(sub_rois, frame, outlines_only=True, linewidth=linewidth)
        overlay.img[sub_roi_overlay.img[:,:,0] != 0] = color

    overlay.to_short()
        
    return overlay

def load_compressed(filename):
    arr = np.load(filename)
    arr = arr['arr_0']
    arr = np.unpackbits(arr, axis=-1, bitorder='little')

    return arr
        
# custom exceptions
class DatatypeException(Exception):
    def __init__(self, message):
        self.message = message
class ChannelException(Exception):
    def __init__(self, message):
        self.message = message
class DimensionException(Exception):
    def __init__(self, message):
        self.message = message
class ImageTypeException(Exception):
    def __init__(self, message):
        self.message = message
