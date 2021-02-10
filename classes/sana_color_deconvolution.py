
import os
import sys
import numpy as np
from numpy.linalg import det, inv

# [0] code from skimage.color
# [1] A Model based Survey of Colour Deconvolution in Diagnostic Brightfield Microscopy: Error Estimation and Spectral Consideration
class StainSeparator:
    def __init__(self, stain_type):

        # define the staining vector
        # TODO: estimate the stain vector, dont just use the default
        if stain_type == 'H-DAB':
            stain_v = np.array([
                [0.65, 0.70, 0.29],
                [0.27, 0.57, 0.78],
                [0.00, 0.00, 0.00],
            ])
        elif stain_type == 'HED':
            stain_v = np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78]
            ])

        # normalize the vectors
        # NOTE: usually the stain vector will already be normalized
        stain_v[0, :] = self.norm(stain_v[0, :])
        stain_v[1, :] = self.norm(stain_v[1, :])
        stain_v[2, :] = self.norm(stain_v[2, :])

        # 3rd color is unspecified, create an orthogonal residual color
        # NOTE: this color will sometimes have negative components, thats okay
        #        since we will check for this later on
        if all(stain_v[2, :] == 0):
            stain_v[2, :] = np.cross(stain_v[0, :], stain_v[1, :])

        # generate the inverse for converting rgb to stain separation
        stain_v_inv = inv(stain_v)

        # store the vectors convert rgb to stain and vice versa
        self.stain_to_rgb = stain_v
        self.rgb_to_stain = stain_v_inv

    def norm(self, v):
        k = (np.sqrt(np.sum(v**2)))
        if k != 0:
            return v / k
        else:
            return v

    def run(self, img, ret=[True, True, True], rescale=False):
        s1, s2, s3 = None, None, None

        # separate the stains, convert from rgb to stains
        stains = self.separate_stains(img)

        # convert each stain channel individually back to a new rgb image
        #  using the original stain vector
        z = np.zeros_like(stains[:, :, 0])
        if ret[0]:
            s1 = self.combine_stains(
                np.stack((stains[:, :, 0], z, z), axis=-1))
        if ret[1]:
            s2 = self.combine_stains(
                np.stack((z, stains[:, :, 1], z), axis=-1))
        if ret[2]:
            s3 = self.combine_stains(
                np.stack((z, z, stains[:, :, 2]), axis=-1))

        if not rescale:
            return s1, s2, s3
        else:
            h = rescale_intensity(stains[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 0], 99)))
            d = rescale_intensity(stains[:, :, 1], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 1], 99)))
            return s1, s2, s3, np.dstack((z, d, h))

    # performs color denconvolution using a rgb image and a inverted stain vector
    def separate_stains(self, rgb):
        v = self.rgb_to_stain

        # scale rgb image from 0 to 1
        rgb = rgb.astype(np.float64) / 255

        # handle log errors
        np.maximum(rgb, 1e-6, out=rgb)
        log_adjust = np.log(1e-6)

        # calculate the optical density, then multiply the inverted stain vector
        stains = (np.log(rgb) / log_adjust) @ v

        # make sure there are no negative stain values
        return np.maximum(stains, 0)

    # converts the stain separated image back to an rgb image using the stain vector
    def combine_stains(self, stain):
        v = self.stain_to_rgb

        # handle log errors
        log_adjust = -np.log(1e-6)

        # generate the rgb image using the stain intensity and the stain vector
        rgb = np.exp(-(stain * log_adjust) @ v)

        # clip values from 0 to 255, then convert to 8 bit ints
        return np.rint(255 * np.clip(rgb, a_min=0, a_max=1)).astype(np.uint8)
#
# end of StainSeparator

#
# end of file
