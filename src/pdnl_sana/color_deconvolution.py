
# system modules
import os
import sys
import numpy as np
from numpy.linalg import det, inv, eig, svd
from matplotlib import pyplot as plt

class StainSeparator:
    """
    Contains methods and attributes to perform Color Deconvolution on IHC images. This is performed using a stain vector which can defined by the stains used in the image, and further tweaked to the image using stain vector estimation.
    [0] code from skimage.color
    [1] A Model based Survey of Colour Deconvolution in Diagnostic Brightfield Microscopy: Error Estimation and Spectral Consideration
    [2] http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    TODO: implement fluorescents
    :param staining_code: stain used on the IHC slides
    :param stain_vector: used to manually define the stain vector -- (9,) array containing the RGB colors (0..1) of each of the up to 3 stains used in the IHC (default: None)
    """    
    def __init__(self, staining_code, stain_vector=None):

        self.stain_vector = StainVector(staining_code, stain_vector)

        # define a series of extreme points based on the digital range of the data
        digital_extrema = np.array(
            [
                [  1,  1,  1],
                [  1,  1,255],
                [  1,255,  1],
                [  1,255,255],
                [255,  1,  1],
                [255,  1,255],
                [255,255,  1],
                [255,255,255]
            ], dtype=np.uint8)[None,:,:]
        
        # find the physical range of each stain in the image
        stains = self.separate(digital_extrema)
        self.min_od = [np.min(stains[:,:,i]) for i in range(3)]
        self.max_od = [np.max(stains[:,:,i]) for i in range(3)]
    #
    # end of constructor

    def to_od(self, img):

        # convert to float, rescale to (0..1)
        # NOTE: clipping to ensure no zeros for log operation
        img_norm = np.clip(img.astype(float), 1, 255) / 255

        # calculate the OD
        # NOTE: clipping to ensure no negatives
        # TODO: why?
        img_od = np.clip(-np.log10(img_norm), 0, None)

        return img_od

    def to_rgb(self, img_od):

        # inverse the OD equation
        img_norm = np.power(10, img_od)

        # rescale to (0..255), convert to uint8
        # NOTE: clipping to ensure (0..1)
        # TODO: why?
        img = np.rint(255 * np.clip(img_norm, 0, 1)).astype(np.uint8)

        return img

    # performs color denconvolution using a rgb image and a inverted stain vector
    def separate(self, img):

        # apply the inverse stain vector to the OD
        # NOTE: clipping to ensure no negatives
        # TODO: why?
        img_sep = self.to_od(img) @ self.stain_vector.v_inv 
        img_sep = np.clip(img_sep, 0, None)  

        return img_sep

    # converts the stain separated image back to an rgb image using the stain vector
    def combine(self, img_sep):

        # generate the rgb image using the stain intensity and the stain vector
        # apply the stain vector to the img to get OD
        img_od = -img_sep @ self.stain_vector.v

        img = self.to_rgb(img_od)

        return img

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
        stain_v = np.array([
            v1,
            v2,
            [0.0, 0.0, 0.0],
        ])
        self.stain_vector.v = stain_v
        self.stain_to_rgb = self.stain_vector.v
        self.rgb_to_stain = self.stain_vector.v_inv        
    
class StainVector:
    """
    Handles the matrix which defines the colors of the given stain, or manually determined stain vector
    :param staining_code: stains used on IHC slides
    :param stain_vector: (9,) array containing the RGB colors (0..1) of each of the up to 3 stains used in the IHC (default: None)
    """    
    def __init__(self, staining_code, stain_vector=None):
        self.staining_code = staining_code

        self.available_stains = {
            'LFB': ['HEM', 'MYE', 'RES'],
            'HE': ['HEM', 'EOS', 'RES'],
            'HED': ['HEM', 'EOS', 'DAB'],
            'H-DAB': ['HEM', 'DAB', 'RES'],
            'CV-DAB': ['CV', 'DAB', 'RES'],
        }
        if not self.staining_code in self.available_stains:
            raise StainNotImplementedError(self.staining_code)
        self.stains = self.available_stains[self.staining_code]

        if not stain_vector is None:
            if len(stain_vector) != 9:
                raise ImproperStainVectorError('Manually defined stain vector must be of shape (9,)')
            self.v = np.array([
                stain_vector[0:3],
                stain_vector[3:6],
                stain_vector[6:9],
            ])
        else:
            if self.staining_code == 'H-DAB':
                self.v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.27, 0.57, 0.78],
                    [0.00, 0.00, 0.00],
                ])  
            elif self.staining_code == 'HE':
                self.v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.00, 0.00, 0.00],
                ])
            elif self.staining_code == 'HED':
                self.v = np.array([
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.27, 0.57, 0.78]
                ])
            elif self.staining_code == 'LFB':
                self.v = np.array([
                    [0.67, 0.73, 0.13],
                    [0.92, 0.38, 0.06],
                    [0.00, 0.00, 0.00],
                ])
            elif self.staining_code == 'CV-DAB':
                self.v = np.array([
                    [0.57, 0.79, 0.22],
                    [0.48, 0.57, 0.67],
                    [0.00, 0.00, 0.00],
                ])
        
        # 3rd channel is unspecified, create an orthogonal residual color
        # NOTE: this color will sometimes have negative components, thats okay since we will check for this later on
        if all(self.v[2, :] == 0):
            self.v[2, :] = np.cross(self.v[0, :], self.v[1, :])
            
        # normalize the vector
        self.v = self.v / np.linalg.norm(self.v, axis=1)

        # store the inverse of the vector
        self.v_inv = inv(self.v)

    def length(self, v):
        return np.sqrt(np.dot(v, v))
    
    def angle(self, i, v2):
        v1 = self.v[i, :]
        return np.arccos(np.dot(v1, v2) / (self.length(v1) * self.length(v2)))

class StainNotImplementedError(Exception):
    def __init__(self, message):
        self.message = message
class ImproperStainVectorError(Exception):
    def __init__(self, message):
        self.message = message
