
# system modules
import os
import sys

# installed modules
import numpy as np
import cv2
import numba

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
def minmax_filter(dist, r, sigma, n_iterations=1, debug=False):

    # make sure image is a float
    img = dist.copy().astype(float)

    # define the disk
    n = 2*r+1
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))
    D[r,r] = 0

    # perform the N iterations of the filter
    for i in range(n_iterations):

        # gauss blur the image
        img_blur = cv2.GaussianBlur(img, (r,r), sigma)

        # perform an iteration of the min - max algorithm        
        img = _minmax(img_blur, D, debug)

        centers = np.where(img == -1)
        if debug:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(img_blur)
            axs[1].imshow(img)
            axs[1].plot(centers[1], centers[0], 'x', color='red')
            axs[0].set_title('iteration=%d' % i)
            plt.show()
    #
    # end of iterations

    return img
#
# end of minmax_filter

