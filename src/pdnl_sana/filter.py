
# system modules
import os
import sys
import ast

# installed modules
import cv2
import numpy as np

import sana.geo

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
            kernel_radius = (kernel_radius,kernel_radius)
        self.kernel_diameter = (2*kernel_radius[0]+1,2*kernel_radius[1]+1)

        self.n_iterations = n_iterations

        self.kernel = cv2.getStructuringElement(self.kernel_type, self.kernel_diameter)
        self.apply = lambda x: cv2.morphologyEx(x, self.filter_type, self.kernel, iterations=self.n_iterations)

    def __str__(self):
        return f"{self.n_iterations} iteration(s) of {self.filter_type_name} filter -- {self.kernel_diameter} {self.kernel_type_name}"

class AnisotropicGaussianFilter:
    def __init__(self, th, sg_x, sg_y):
        n = int(round(sg_y*6))
        if n % 2 == 0:
            n += 1
        self.kernel = np.zeros((n, n), dtype=float)
        for j in range(n):
            y = j - n // 2
            for i in range(n):
                x = i - n // 2
                self.kernel[j,i] = (1/(2*np.pi*sg_x*sg_y)) * \
            np.exp(-( ( (x * np.cos(th) + y * np.sin(th))**2/(sg_x)**2 ) + ( (-x*np.sin(th) + y*np.cos(th))**2/(sg_y)**2 ) )/2)

        #self.apply = lambda x: signal.convolve2d(x, self.kernel, mode='same') / np.sum(self.kernel)
        self.apply = lambda frame, stride: frame.convolve(self.kernel, stride, align_center=True)

def get_gaussian_kernel(length, sigma):
    """
    Builds a 2D gaussian kernel
    :param length: side length of the square kernel image
    :param sigma: standard deviation of the gaussian
    :returns: 2D image array
    """
    
    # calculate the 1D gaussian
    x = np.linspace(-(length-1)/2, (length-1)/2, length)
    g1 = np.exp(-0.5 * np.square(x) / np.square(sigma))

    # get the 2D gaussian and normalize
    g2 = np.outer(g1,g1)
    g2 = g2 / np.sum(g2)

    return g2
