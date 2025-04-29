
# system packages
import os

# installed packages
import cv2
import numpy as np
from PIL import Image
import shapely.geometry
from matplotlib import colors
from matplotlib import pyplot as plt

# sana packages
import sana.geo
import sana.filter

# TODO: check parameter usage
class Frame:
    """ 
    Provides an interface for applying a series of image processing functions to the stored image array. This class attempts to perform operations in place so that memory usage does not explode.
    
    :param img: (M,N,d) array
    :param level: the slide level the array was loaded from
    :param converter: Converter object used to manipulate units
    :param padding: amount of padding added to the frame
    :param is_deformed: whether or not this frame was normalized to a common cortical space, usually False
    """
    def __init__(self, img, level=None, converter=None, padding=0, is_deformed=False):

        # load/store the image array
        if type(img) is str:
            if img.endswith('.npz'):
                self.img = load_compressed_array(img)
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
        self.level = level
        self.converter = converter
        self.padding = padding
        self.is_deformed = is_deformed
        
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
    def is_binary(self):
        """
        Checks if the image has only 0's and 1's in it
        """
        return self.is_gray() and self.is_short() and (np.max(self.img) <= 1) and (np.min(self.img) >= 0)
    def is_short(self):
        return self.img.dtype == np.uint8
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

    def get_histogram(self, mask=None):
        """
        Generates a 256 bin histogram for each color channel
        """
        if not self.is_short():
            raise DataTypeException("Histogram must be calculated on short pixel values")
        
        histogram = np.zeros((256, self.img.shape[-1]))
        for i in range(histogram.shape[-1]):
            if mask is None:
                histogram[:,i] = np.histogram(self.img[:,:,i], bins=256, range=(0,256))[0]
            else:
                histogram[:,i] = np.histogram(self.img[:,:,i][mask.img[:,:,0] != 0], bins=256, range=(0,256))[0]
        return histogram

    def get_tile(self, loc, size):
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
            pad_x1 = x1-w
        if y1 >= h:
            pad_y1 = y1-h

        x0 = np.clip(x0, 0, None)
        y0 = np.clip(y0, 0, None)
        x1 = np.clip(x1, None, w)
        y1 = np.clip(y1, None, h)
            
        img = self.img[y0:y1, x0:x1, :]
        img = np.pad(img, [
            [pad_y0, pad_y1],
            [pad_x0, pad_x1],
            [0,0],
        ])
        return img

    def crop(self, loc, size):
        """
        Crops the image to the given rectangle
        """
        self.img = self.get_tile(loc, size)

    def resize(self, size, interpolation=cv2.INTER_NEAREST):
        """
        Resizes the image array to the specified size
        """
        if not self.converter is None:
            size = self.converter.to_pixels(size, self.level)
            size = self.converter.to_int(size)
        w, h = size
        self.img = cv2.resize(self.img, dsize=(w, h), interpolation=interpolation)
        self.check_channels()

    def apply_morphology_filter(self, morphology_filter):
        if not self.is_binary():
            raise ImageTypeException("Cannot apply morphology filter to non-binary image")
        self.img = morphology_filter.apply(self.img)[:,:,None]

    def convolve(self, kernel: np.ndarray, tile_step: sana.geo.Point, align_center=False):
        """
        Performs a (downsampled) convolution of the given kernel over the image. This is done by creating tile views into the frame, then applying the kernel to each of the views
        :param kernel: (M,N) array to apply to each tile
        :param tile_step: stride distance between tiles
        :param align_center: if True, pads the frame to center align rather than upper left align
        """
        if not self.is_gray():
            raise ImageTypeException('Convolution only supported on single-channel images')

        tile_size = sana.geo.point_like(tile_step, kernel.shape[1], kernel.shape[0])
        
        # generate the tile views to convolve the kernel with
        tiles = self.to_tiles(tile_size, tile_step, align_center=align_center)

        # perform the convolution
        result = np.sum(tiles * kernel[None,None,:,:], axis=(2,3))

        # TODO: do we always want to normalize?
        ones = frame_like(self, np.ones_like(self.img))
        ones_tiles = ones.to_tiles(tile_size, tile_step, align_center=align_center)
        norm_factor = np.sum(ones_tiles * kernel[None,None,:,:], axis=(2,3))
        result = result.astype(float) / norm_factor

        return result

    def remove_background(self, min_background=0, max_background=255, debug=False):
        """
        Performs a background subtraction process on a grayscale image. The process works by estimating a background image, which is then subtracted from the original image. The background is found by looking at a specified range of pixel values along with convolving a gaussian kernel over the image. Note that background in this case usually refers to non-specific staining data as opposed to glass slide background. It is trivial to remove slide background with a simple threshold, but non-specific staining background can vary throughout the image so some type of adaptive thresholding is necessary. This functions acts somewhat like an adaptive threshold, since we subtract a variable background value throughout the image.

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

        # get the tiles for the convolution
        # TODO: check the parameters, make these arg
        tsize = self.converter.to_int(self.converter.to_pixels(
            sana.geo.Point(100, 100, is_micron=True), 
            self.level
        ))
        if tsize[0] % 2 == 0: tsize[0] += 1
        if tsize[1] % 2 == 0: tsize[1] += 1
        tstep = sana.geo.Point(50, 50, is_micron=True)
        kern_sigma = tsize[0]/5
        kernel = sana.filter.get_gaussian_kernel(tsize[0], kern_sigma)

        # find the background simply by gaussian blurring the image
        # NOTE: used to support masking by a minimum+maximum background value but this is not currently used
        background_image = self.convolve(kernel, tstep, align_center=True)

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

        return background_frame

    def rotate(self, angle, interpolation=cv2.INTER_LINEAR):
        """
        Rotates the image around it's center
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
        self.check_channels()

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
            if self.is_float() or fpath.endswith('.npy'):
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

    def mask(self, mask_frame, mask_value=0, invert=False):
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
        if invert:
            neg_value = 1
        else:
            neg_value = 0
        self.img[mask_frame.img[:,:,0] == neg_value] = mask_value

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
    
    def threshold(self, threshold, x=0, y=1):
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

    def to_tiles(self, tsize: sana.geo.Point, tstep: sana.geo.Point, align_center=False):
        """
        Uses stride_tricks to create views into the array which saves memory and processing time
        :param tsize: size of tile
        :param tstep: stride length between tiles
        :param align_center: if True, pads the frame so that the tiles are center aligned rather than upper left aligned
        """
        if not self.is_gray():
            raise ChannelException(f"Must be a 1 channel image to tile -- shape={self.img.shape}")
        
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
        if align_center:
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

def frame_like(frame, img):
    """
    Creates a Frame object with the same parameters as the given frame
    :param frame: frame to emulate
    :param img: image array to store in the new Frame object
    """
    return Frame(img,
                 level=frame.level,
                 converter=frame.converter,
                 padding=frame.padding,
                 is_deformed=frame.is_deformed)
#
# end of frame_like

def create_mask_like(frame: Frame, polygons: [sana.geo.Polygon]):
    """
    Creates a mask with the same size as the given Frame using a list of Polygons
    :param frame: frame to emulate
    :param polygons: list of Polygons to fill in
    """
    int_polygons = []
    for p in polygons:
        frame.converter.to_pixels(p, frame.level)
        p = frame.converter.to_int(p).connect()
        int_polygons.append(p)

    mask_img = np.zeros(frame.img.shape[:2], dtype=np.uint8)
    mask_img = cv2.fillPoly(mask_img, pts=int_polygons, color=1)

    return frame_like(frame, mask_img)

def load_compressed_array(filename):
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
