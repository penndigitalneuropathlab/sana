
import os
import sys
import numpy as np
from numpy.linalg import det, inv, eig, svd

# [0] code from skimage.color
# [1] A Model based Survey of Colour Deconvolution in Diagnostic Brightfield Microscopy: Error Estimation and Spectral Consideration
class StainSeparator:
    def __init__(self, stain_type, stain_target, stain_v=None, od=False):

        # define the staining vector
        # TODO: estimate the stain vector, dont just use the default
        if stain_type == 'H-DAB':
            if stain_v is None:
                stain_v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.27, 0.57, 0.78],
                    [0.00, 0.00, 0.00],
                ])
            else:
                stain_v = np.array(stain_v)
            self.stains = ['HEM', 'DAB', 'RES']

        elif stain_type == 'HED':
            if stain_v is None:
                stain_v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.27, 0.57, 0.78]
                ])
            else:
                stain_v = np.array(stain_v)
            self.stains = ['HEM', 'EOS', 'DAB']

        # define which stains to return
        if stain_target.upper() == 'ALL':
            self.ret = [True, True, True]
        else:
            self.ret = list(map(lambda x: x == stain_target.upper(),
                                self.stains))
        self.od = od

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

    def run(self, img, rescale=False):
        s1, s2, s3 = None, None, None

        # separate the stains, convert from rgb to stains
        stains = self.separate_stains(img)

        if self.od:
            s1, s2, s3 = stains[:, :, 0], stains[:, :, 1], stains[:, :, 2]
        else:
            # convert each stain channel individually back to a new rgb image
            #  using the original stain vector
            z = np.zeros_like(stains[:, :, 0])
            s1 = self.combine_stains(
                np.stack((stains[:, :, 0], z, z), axis=-1))
            s2 = self.combine_stains(
                np.stack((z, stains[:, :, 1], z), axis=-1))
            s3 = self.combine_stains(
                np.stack((z, z, stains[:, :, 2]), axis=-1))

        ret = []
        if self.ret[0]:
            ret.append(s1)
        if self.ret[1]:
            ret.append(s2)
        if self.ret[2]:
            ret.append(s3)

        if rescale:
            h = rescale_intensity(stains[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 0], 99)))
            d = rescale_intensity(stains[:, :, 1], out_range=(0, 1),
                      in_range=(0, np.percentile(stains[:, :, 1], 99)))
            ret.append(np.dstack((z, d, h)))

        return ret

    # NOTE: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    def estimate_stain_vector(self, img):
        mx, mi = 1.0, 0.001
        alpha = 1 / 100.0

        # flatten the image
        img = img.reshape(img.shape[0]*img.shape[1], 3)

        # convert to od
        od = self.to_od(img)

        # remove data too faintly or densely stained
        mag = np.sum(od**2, axis=-1)
        od = od[np.where(np.logical_and(mag < mx, mag >= mi))]

        # perform singular value decomposition, find 2 SVD direction vectors with the highest singular values
        cov = np.cov(od, rowvar=False)
        w, v = eig(cov)
        inds = np.argsort(w)
        v1 = v[:, inds[-1]]
        v2 = v[:, inds[-2]]

        # find the angle of each pixel, find the min and max extrema
        phi = np.arctan2(od @ v1, od @ v2)
        inds = np.argsort(phi)
        ind1 = inds[int(alpha*phi.shape[0])]
        ind2 = inds[int((1-alpha)*phi.shape[0])]

        # construct the stain vector
        stain_vector = [
            list(od[ind1]),
            list(od[ind2]),
            [0.0, 0.0, 0.0],
        ]
        return stain_vector

    def to_od(self, rgb):
        rgb = rgb.astype(np.float64) / 255
        np.maximum(rgb, 1e-6, out=rgb)
        log_adjust = np.log10(1e-6)
        return np.log10(rgb) / log_adjust

    # performs color denconvolution using a rgb image and a inverted stain vector
    def separate_stains(self, rgb):
        v = self.rgb_to_stain

        od = self.to_od(rgb)
        stains = od @ v
        return np.maximum(stains, 0)

        # # scale between 0 and 1, ensure there are no zeros in the analysis
        # rgb = np.maximum(rgb, 1).astype(np.float64) / 255
        #
        # # calculate optical density, ensure 0 is not sent to the log10 function
        # od = np.maximum(0, -np.log10(rgb))
        #
        # # apply the stain vector to the OD
        # stains = od @ v

        # make sure there are no negative stain values
        return stains

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
