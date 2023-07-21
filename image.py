
# system packages
import os
import sys

# installed packages
import cv2
import numpy as np
from PIL import Image
import nibabel as nib

# sana packages
from sana_geo import Point, point_like

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

# TODO: check parameter usage
class Frame:
    """ Provides an interface for applying a series of image processing functions to the stored 2D numpy image array. This class attempts to perform operations in place so that memory usage does not explode.
    :param img: the 2d numpy array to store
    :param level: the slide level the array was loaded from
    :param converter: Converter object to use to manipulate units
    :param slide_threshold: grayscale intensity value to remove background slide data
    :param slide_color: color of slide background to use when needed
    :param padding: amount of padding added to the frame
    """
    def __init__(self, img, level=-1, converter=None, slide_threshold=None, slide_color=None, padding=0):

        # load/store the image array
        if type(img) is str:
            self.img = np.array(Image.open(img))
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
        return Point(w, h, level=self.level)

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

    def is_short(self):
        """
        Checks if the image is in floating point values
        """
        return self.img.dtype != np.uint8

    def is_binary(self):
        """
        Checks if the image has only 0's and 1's in it
        """
        return self.is_gray() and self.is_short() and (np.max(self.img) > 1)

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

    # TODO: add padding??
    def get_tile(self, rect):
        """
        Gets a cropped rectangular tile from the frame
        :param rect: Rectangle object
        """
        self.converter.to_pixels(rect, self.level, round=True)
        w, h = self.size()
        (x0, y0) = rect.loc[0], rect.loc[1]
        (x1, y1) = x0+rect.size[0], y0+rect.size[1]
        x0 = np.clip(x0, 0, None)
        y0 = np.clip(y0, 0, None)
        x1 = np.clip(x1, None, w)
        y1 = np.clip(y1, None, h)
        return self.img[y0:y1, x0:x1, :]

    def crop(self, rect):
        """
        Crops the image to the given rectangle
        """
        self.img = self.get_tile(rect)

    def resize(self, size, interpolation=cv2.INTER_NEAREST):
        """
        Resizes the image array to the specified size
        """
        size = self.converter.to_pixels(size, self.level, round=True)
        w, h = size
        self.img = cv2.resize(self.img, dsize=(w, h), interpolation=interpolation)
        self.check_channels()

    # TODO: maybe look at histogram to auto-set the min/max values?
    def remove_background(self, min_background=1, max_background=255, debug=False):
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
            fig, axs = plt.subplots(1,3, figsize=(20,20))
            axs[0].imshow(self.img)

        # TODO: check this, or maybe make it an argument
        # run this function at a lower resolution to make it run faster
        level = 1
        ds = self.converter.ds[level] / self.converter.ds[self.level]
        ds_size = self.size() / ds # TODO: is this int?

        # we will tile the image during the convolution with the gaussian kernel
        # TODO: check these params, make them arguments?
        tsize = Point(200, 200, is_micron=True)
        tstep = Point(10, 10, is_micron=True)
        tiler = Tiler(level, self.converter, tsize, tstep, fpad_mode='constant')

        # run the downsampling
        ds_frame = self.copy()
        ds_frame.resize(ds_size, interpolation=cv2.INTER_CUBIC) # TODO: check interpolation
        tiler.set_frame(frame_ds)

        # load the tiles
        tiles = tiler.load_tiles()
        tds = tiler.ds

        # image array that stores the background values for each tile
        background_image = np.zeros([tiles.shape[0], tiles.shape[1]], dtype=float)

        # 2D gaussian kernel to apply to the tiles
        kern_length = tiles[0][0].shape[0]
        kern_sigma = 1.0 # TODO: make this a parameter
        kern = gaussian_kernel(kern_length, kern_sigma)

        # start the convolution
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):

                # get the pixels within the value range in the tile
                tile = tiles[i][j]
                valid = (data >= min_background) & (data <= max_background)

                # background is calculated as the sum of valid pixels after applying the gaussian kernel
                background_value = np.sum((tile * kern)[valid])
                background_image[i][j] = background_value

        # resize the background frame back to original resolution
        background_frame = frame_like(self, background_image)
        background_frame.resize(self.size())

        # finally, subtract the background
        # NOTE: we clip to 0 here, this is okay since this will be background data which we don't care about
        self.to_float()
        self.img = self.img - background_frame.img
        np.clip(self.img, 0, 255, out=self.img)
        self.to_short()

        if debug:
            axs[1].imshow(background_frame.img)
            axs[2].imshow(self.img)

    def rotate(self, angle):
        """
        Rotates the image TODO around it's center
        :param angle: rotation angle in degrees
        """
        # calculate the rotation matrix, along with the new width and height of the image
        M, w, h = self.get_rotation_matrix(angle)

        # perform the affine transformation
        self.warp_affine(M, w, h)

    def get_rotation_matrix(self, angle):
        """
        Finds the rotation matrix for the given angle and image center, also calculates the new width and height of the rotated image
        :param angle: rotation angle in degrees
        """

        # get the rotation point
        w, h = self.size()
        cx, cy = w//2, h//2

        # find the rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # compute the new dimensions
        cos_term = np.abs(M[0,0])
        sin_term = np.abs(M[0,1])
        new_w = int((h * sin_term) + (w * cos_term))
        new_h = int((h * cos_term) + (w * sin_term))

        # translate the rotation matrix so that the image center doesn't move
        M[0,2] += (new_w / 2) - cx
        M[1,2] += (new_h / 2) - cy

        return M, new_w, new_h
    
    def warp_affine(self, M, w, h):
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
            after = point_like(pad, (0,0))
        elif alignment == 'center':
            before = pad//2
            after = pad - before
        elif alignment == 'right':
            before = point_like(pad, (0,0))
            after = pad
        self.img = np.pad(self.img, ((before[1], after[1]), # y
                                     (before[0], after[1]), # x
                                     (0, 0),                # c
                                     ), mode=mode)
        
    def save(self, fname, invert_sform=False, spacing=None):
        """
        Writes the image array to a file
        :param fname: filename to write to
        :param invert_sform: see sana.image.save_nifti
        :param spacing: see sana.image.save_nifti
        """
        if fname.endswith('.nii.gz'):
            self.save_nifti(fname, invert_sform, spacing)
        else:
            # can't write png's etc as floats, just store the image array
            if self.is_float():
                np.save(os.path.splitext(fname)[0]+'.npy', self.img)
            else:
                im = self.img
                if self.is_gray():
                    im = self.img[:,:,0]
                if self.is_binary():
                    im *= 255
                im = Image.fromarray(im)
                im.save(fname)
    
    def save_nifti(self, fname, invert_sform=False, spacing=None):
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
            spacing = [self.converter.ds[self.lvl] * (self.converter.mpp / 1000)] * 2

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
        nib.save(nii, fname)
  
    def mask(self, mask_frame, mask_value=0):
        """
        Applies a binary mask to the image
        :param mask_frame: a binary image frame with the same dimensions
        :param mask_value: value to set the masked data to
        """
        if self.size() != mask_frame.size():
            raise DimensionException('Mask image {} has different dimensions than the frame {}'.format(mask_frame.size(), self.size()))
        
        if self.is_rgb() and type(mask_value) is int:
            mask_value = (mask_value, mask_value, mask_value)
        self.img[mask_frame.img[:,:,0] == 0] = mask_value

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
        

def gaussian_kernel(length, sigma):
    """
    Builds a 2D gaussian kernel
    :param length: side length of the square kernel image
    :param sigma: standard deviation of the gaussian
    :returns: 2D image array
    """

    # calculate the 1D gaussian
    x = np.linspace(-(l-1)/2, (l-1)/2, l)
    g1 = np.exp(-0.5 * np.square(x) / np.square(sg))

    # get the 2D gaussian and normalize
    g2 = np.outer(g1,g1)
    g2 = g2 / np.sum(g2)

    return g2
