
# system modules
import os
import sys
import ast

# installed modules
import cv2
import numpy as np

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
