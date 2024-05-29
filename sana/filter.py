
# system modules
import os
import sys
import ast

# installed modules
import numpy as np
import cv2
import numba
import cv2
from scipy import signal

# debugging modules
from matplotlib import pyplot as plt

# C optimized internal function to perform the actual filter
@numba.jit(nopython=True)
def _minmax(img, D, debug=False):

    # get the number of pixels in the disk
    n = D.shape[0]
    r = (n - 1) // 2
    N_D = np.sum(D)

    # this holds the result of the filter, representing the proportion of
    # pixels above and below any given pixel in the image
    I_e = np.zeros(tuple(img.shape), dtype=numba.float32)

    # loop through all pixels, o
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):

            # get the bounds for the patch centered on o
            x0, x1 = i-r, i+r+1
            y0, y1 = j-r, j+r+1

            # clip the bounds to the image boundaries as necessary
            cx0 = x0 if x0 >= 0 else 0
            cx1 = x1 if x1 < img.shape[1] else img.shape[1]-1
            cy0 = y0 if y0 >= 0 else 0
            cy1 = y1 if y1 < img.shape[0] else img.shape[0]-1

            # extract the patch and the pixel
            o = img[j, i]
            patch = np.zeros((n,n), dtype=numba.float32)
            patch[cy0-y0:n+cy1-y1, cx0-x0:n+cx1-x1] = img[cy0:cy1, cx0:cx1]

            # get the number of pixel intensities above and below pixel o
            above_o = 0
            below_o = 0
            for Dj in range(D.shape[0]):
                for Di in range(D.shape[1]):
                    if D[Dj,Di] == 1:
                        if patch[Dj,Di] < o:
                            below_o += 1
                        else:
                            above_o += 1
            #
            # end of pixel counting

            # calculate result of the filter at pixel o
            I_e[j, i] = (above_o - below_o) / N_D
    #
    # end of o loop

    return I_e
#
# end of _minmax

# IMPORTANT: This function expects a distance transform image
# 
# this is a wrapper function for _minmax, it prepares the arrays
#  and performs the general blurring and iterations
#
# this function calculates the number of pixels in a disk "D" centered on a
#   pixel "o" that are both above or below the intensity of pixel o
# namely, it calculates the following equation for each pixel
#   o' = (above - below) / N
#   where "above" is number of pixels w/ intensities greater than or equal to o
#         "below" is number of pixels w/ intensities less than o
#     and N is number of pixels in the disk
# NOTE: if all pixels are less than o, we get 0-N/N = -1
#         and all pixels greater than or equal to o we get N-0/N = 1
#       the background of the image goes to 1, as all surrounding pixels equal 0
#       the centers of cells go to -1, as all surrounding pixels are less than the center
def min_max_filter(frame, img, r, n_iterations=1, debug=False):

    # define the disk
    n = 2*r+1
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))
    D[r,r] = 0

    if debug:
        fig, axs = plt.subplots(1,1+2*n_iterations, sharex=True, sharey=True)
        axs[0].imshow(frame.img)

    # perform the N iterations of the filter
    for i in range(n_iterations):

        # gauss blur the image
        k = int(r//2)
        if k % 2 == 0:
            k += 1
        img_blur = cv2.GaussianBlur(img, (2*k+1,2*k+1), 0)

        # perform an iteration of the min - max algorithm        
        img = -1 * _minmax(img_blur, D, debug)

        centers = np.where(img == 1)
        if debug:
            axs[1+2*i].imshow(img_blur)
            axs[1+2*i].set_title('iteration=%d' % i)
            axs[1+2*i+1].imshow(img)
            axs[1+2*i+1].plot(centers[1], centers[0], 'x', color='red')
    #
    # end of iterations

    return img
#
# end of minmax_filter

class AnisotropicGaussianFilter:
    def __init__(self, th, sg_x, sg_y):
        n = int(round(sg_y*3))
        self.kernel = np.zeros((n, n), dtype=float)
        for j in range(n):
            y = j - n // 2
            for i in range(n):
                x = i - n // 2

                self.kernel[j,i] = (1/(2*np.pi*sg_x*sg_y)) * \
            np.exp(-( ( (x * np.cos(th) + y * np.sin(th))**2/(sg_x)**2 ) + ( (-x*np.sin(th) + y*np.cos(th))**2/(sg_y)**2 ) )/2)
                
        self.apply = lambda x: signal.convolve2d(x, self.kernel, mode='same') / np.sum(self.kernel)


class MorphologyFilter:
    """
    Wrapper for OpenCV's morphology filters
    """
    NAME_TO_FILTER = {
        'erosion': cv2.MORPH_ERODE,
        'dilation': cv2.MORPH_DILATE,
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
    }
    NAME_TO_KERNEL = {
        'ellipse': cv2.MORPH_ELLIPSE,
        'rectangle': cv2.MORPH_RECT,
    }
    def __init__(self, filter_type, kernel_type, kernel_radius, n_iterations=1):

        self.filter_type_name = filter_type
        self.filter_type = self.NAME_TO_FILTER[self.filter_type_name]

        self.kernel_type_name = kernel_type
        self.kernel_type = self.NAME_TO_KERNEL[self.kernel_type_name]

        if type(kernel_radius) is int:
            self.kernel_diameter = (2*kernel_radius+1, 2*kernel_radius+1)
        else:
            kernel_radius = ast.literal_eval(kernel_radius)
            self.kernel_diameter = (2*kernel_radius[0]+1,2*kernel_radius[1]+1)

        self.n_iterations = n_iterations

        self.kernel = cv2.getStructuringElement(self.kernel_type, self.kernel_diameter)

        self.apply = lambda x: cv2.morphologyEx(x, self.filter_type, self.kernel, iterations=self.n_iterations)

    def __str__(self):
        return f"{self.n_iterations} iteration(s) of {self.filter_type_name} filter -- {self.kernel_diameter} {self.kernel_type_name}"