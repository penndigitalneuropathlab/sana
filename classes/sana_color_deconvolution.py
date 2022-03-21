
import os
import sys
import numpy as np
from numpy.linalg import det, inv, eig, svd

# [0] code from skimage.color
# [1] A Model based Survey of Colour Deconvolution in Diagnostic Brightfield Microscopy: Error Estimation and Spectral Consideration
class StainSeparator:
    def __init__(self, stain_type, stain_target,
                 stain_vector=None, od=False, gray=True):

        # define the stain vector
        self.stain_type = stain_type
        self.stain_target = stain_target
        self.stain_vector = StainVector(
            self.stain_type, self.stain_target, stain_vector)
        self.stain_to_rgb = self.stain_vector.v
        self.rgb_to_stain = self.stain_vector.v_inv

        # define which stains to return
        if type(self.stain_target) is list:
            self.ret = self.stain_target
        elif self.stain_target.upper() == 'ALL':
            self.ret = [True, True, True]
        else:
            self.ret = list(map(lambda x: x == self.stain_target.upper(),
                                self.stain_vector.stains))

        # parameters defining the type of image to generate
        self.od = od
        self.gray = gray
    #
    # end of constructor

    def run(self, img):
        s1, s2, s3 = None, None, None

        # separate the stains, convert from rgb to stains
        stains = self.separate_stains(img)
        s1, s2, s3 = stains[:, :, 0], stains[:, :, 1], stains[:, :, 2]

        # TODO: make sure this is working properly, check the histograms
        # convert each stain channel back to a new rgb image
        #  using the original stain vector
        if not self.od:
            z = np.zeros_like(s1)
            s = np.stack((z,z,z), axis=-1)
            if self.ret[0]:
                s[:,:,0] = s1
            if self.ret[1]:
                s[:,:,1] = s2
            if self.ret[2]:
                s[:,:,2] = s3
            s = self.combine_stains(s)

            # convert to grayscale
            if self.gray:
                s = np.rint(np.dot(s.astype(float),
                    [0.2989, 0.5870, 0.1140])).astype(np.uint8)
            #
            # end of to grayscale processing
        #
        # end of to rgb processing

        return s
    #
    # end of run

    # NOTE: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    def estimate_stain_vector(self, img):
        mx, mi = 1.0, 0.05
        alpha = 1 / 100.0

        # flatten the image
        img = img.reshape(img.shape[0]*img.shape[1], 3)

        # convert to od
        od = self.to_od(img)

        # remove data that is too faint
        od = od[np.where(np.logical_or(
            od[:, 0] >= mi, od[:, 1] >= mi, od[:, 2] >= mi))]

        # remove data that is too dense
        mag = np.sum(od**2, axis=-1)
        od = od[np.where(mag <= mx**2)]

        # perform singular value decomposition, find 2 SVD direction vectors with the highest singular values
        cov = np.cov(od.T)
        w, v = eig(cov)
        inds = np.argsort(w)
        v1 = v[:, inds[-1]]
        v2 = v[:, inds[-2]]

        # find the angle of each pixel, find the min and max extrema
        phi = np.arctan2(od @ v1, od @ v2)
        inds = np.argsort(phi)
        ind1 = inds[int(alpha*phi.shape[0])]
        ind2 = inds[int((1-alpha)*phi.shape[0])]
        v1 = od[ind1]
        v2 = od[ind2]

        # make sure the vectors are in the expected order
        #  i.e. it should be v1=HEM, v2=DAB, sometimes it's switched
        a11 = self.stain_vector.angle(0, v1)
        a12 = self.stain_vector.angle(0, v2)
        a21 = self.stain_vector.angle(1, v1)
        a22 = self.stain_vector.angle(1, v2)
        if min(a12, a21) < min(a11, a22):
            v = v2.copy()
            v2 = v1.copy()
            v1 = v

        # construct the new stain vector
        stain_v = [
            v1,
            v2,
            [0.0, 0.0, 0.0],
        ]
        self.stain_vector = StainVector(
            self.stain_type, self.stain_target, stain_v)
        self.stain_to_rgb = self.stain_vector.v
        self.rgb_to_stain = self.stain_vector.v_inv

    def to_od(self, rgb):

        # scale the data by the max, ensure no zeros
        rgb = np.clip(rgb.astype(np.float64), 1, 255) / 255

        # get the OD, ensure no negatives
        return np.clip(-np.log10(rgb), 0, None)

    # performs color denconvolution using a rgb image and a inverted stain vector
    def separate_stains(self, rgb):
        v = self.rgb_to_stain

        # apply the stain vector to the OD, ensure no negative stain values
        od = self.to_od(rgb)
        stains = od @ v
        return np.clip(stains, 0, None)

    # converts the stain separated image back to an rgb image using the stain vector
    def combine_stains(self, stain):
        v = self.stain_to_rgb

        # generate the rgb image using the stain intensity and the stain vector
        rgb = np.power(10, -stain @ v)
        # rgb = np.exp(-stain @ v)

        # clip values from 0 to 255, then convert to 8 bit ints
        return np.rint(255 * np.clip(rgb, 0, 1)).astype(np.uint8)
#
# end of StainSeparator

class StainVector:
    def __init__(self, stain_type, stain_target, stain_vector=None):
        self.stain_type = stain_type
        self.stain_target = stain_target
        if self.stain_type == 'H-DAB':
            if stain_vector is None:
                self.v = stain_v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.27, 0.57, 0.78],
                    [0.00, 0.00, 0.00],
                ])
            else:
                self.v = np.array(stain_vector)
            self.stains = ['HEM', 'DAB', 'RES']
        if self.stain_type == 'HED':
            if stain_vector is None:
                self.v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.27, 0.57, 0.78]
                ])
            else:
                self.v = np.array(stain_vector)
            self.stains = ['HEM', 'EOS', 'DAB']

        # 3rd color is unspecified, create an orthogonal residual color
        # NOTE: this color will sometimes have negative components, thats okay
        #        since we will check for this later on
        if all(self.v[2, :] == 0):
            self.v[2, :] = np.cross(self.v[0, :], self.v[1, :])

        # normalize the vector
        self.v[0, :] = self.norm(self.v[0, :])
        self.v[1, :] = self.norm(self.v[1, :])
        self.v[2, :] = self.norm(self.v[2, :])

        # store the inverse of the vector
        self.v_inv = inv(self.v)

    def norm(self, v):
        k = (np.sqrt(np.sum(v**2)))
        if k != 0:
            return v / k
        else:
            return v
    def length(self, v):
        return np.sqrt(np.dot(v, v))

    def angle(self, i, v2):
        v1 = self.v[i, :]
        return np.arccos(np.dot(v1, v2) / (self.length(v1) * self.length(v2)))


#
# end of file
