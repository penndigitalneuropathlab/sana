
# system packages
import os
import sys

# installed packages
import cv2
import numpy as np
from PIL import Image
from sana_geo import Point

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
