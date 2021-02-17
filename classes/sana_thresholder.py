
import os
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sana_color_deconvolution import StainSeparator
class Thresholder:
    def __init__(self, data, k, n):
        self.data = data
        self.k = k
        self.n = n

    def run_gmm(self):
        self.gmm()
        self.mle()

    # generates the means and stdevs of the given data and pre-defined k value
    def gmm(self):
        gmm = GaussianMixture(
            n_components=self.k, covariance_type='full').fit(self.data)

        # get the sorted variances and corresponding means
        inds = gmm.covariances_[:, 0, 0].argsort()
        means = gmm.means_[:, 0][inds]
        vars = gmm.covariances_[:, 0, 0][inds]

        # take the top N distributions
        means = np.rint(means[:self.n]).astype(int)
        vars = vars[:self.n]

        # sort the distributionsn by the means
        inds = means.argsort()
        self.means = means[inds]
        self.vars = vars[inds]

    # finds the optimal boundaries between a set of means and vars
    def mle(self):

        # perform maximum likelihood estimation
        # NOTE: this is finding the crossing of the PDFs between the means
        t = np.arange(0, 256, 1)
        thresholds = []
        for i in range(self.n-1):

            # generate the pdfs
            p1 = multivariate_normal(self.means[i], self.vars[i]).pdf(t)
            p2 = multivariate_normal(self.means[i+1], self.vars[i+1]).pdf(t)

            # only evaluate the pdfs between the 2 means
            p1 = p1.flatten()[self.means[i]:self.means[i+1]]
            p2 = p2.flatten()[self.means[i]:self.means[i+1]]

            # find the last x value where p1 is more probable than p2
            thresholds.append(np.nonzero(p1 > p2)[0][-1] + self.means[i])

        self.thresholds = thresholds

    # defines the threshold as 1 std above the mean
    def close_right(self):
        thresholds = []
        for i in range(self.n-1):
            thresholds.append(self.means[i] + 2*np.sqrt(self.vars[i]))
        self.thresholds = thresholds
#
# end of Thresholder

class TissueThresholder(Thresholder):
    def __init__(self, frame, blur=5, mi=0, mx=255):
        self.frame = frame
        self.mi = mi
        self.mx = mx

        # convert to grayscale and blur
        self.frame.to_gray()
        self.frame.gauss_blur(blur)

        # flatten the data
        # TODO: downsampling the data is not a great solution...
        data = self.frame.img.flatten()
        data = data[data >= self.mi]
        data = data[data <= self.mx]
        data = data[::500]
        data = data[:, None]

        # initialize the thresholder
        super().__init__(data, 8, 2)

    def mask_frame(self):

        # run the gmm algorithm to define the means and vars of the data
        self.gmm()

        # use mle to define the crossing of PDFs
        self.mle()
        self.tissue_threshold = self.thresholds[0]

        # threshold the frame to generate a tissue mask
        self.frame.threshold(self.tissue_threshold, x=1, y=0)
#
# end of TissueThresholder

class StainThresholder(Thresholder):
    def __init__(self, frame, blur=0, mi=0, mx=255):
        self.frame = frame
        self.mi = mi
        self.mx = mx

        # convert to grayscale and blur
        self.frame.to_gray()
        self.frame.gauss_blur(blur)

        # flatten the data
        data = self.frame.img.flatten()
        data = data[data >= self.mi]
        data = data[data <= self.mx]
        data = data[::1000]
        data = data[:, None]

        # initialize the thresholder
        super().__init__(data, 3, 3)

    def mask_frame(self):

        # run the gmm algorithm to define the means and vars of the data
        self.gmm()

        # use mle to define the crossing of PDFs
        self.mle()
        self.stain_threshold, self.tissue_threshold = \
            self.thresholds

        # threshold the frame to generate a tissue mask
        self.frame.threshold(
            self.tissue_threshold, x=np.copy(self.frame.img), y=255)
        self.frame.threshold(
            self.stain_threshold, x=0, y=np.copy(self.frame.img))

#
# end of StainThresholder

class CellThresholder(Thresholder):
    def __init__(self, frame, tissue_mask, blur=0):
        self.frame = frame

        self.frame.to_gray()
        self.frame.gauss_blur(blur)

        # mask out any slide background
        self.frame.mask(tissue_mask, value=255)
        self.frame.mask_histo = self.frame.histogram()

        # flatten the data and remove masked data from the analysis
        data = self.frame.img[self.frame.img != 255].flatten()[:, None]

        # initalize the thresholder
        super().__init__(data, 2, 2)

    def mask_frame(self):

        # run the gmm algorithm to define the means and vars of the data
        self.gmm()

        # define threshold as the crossing between PDFs
        self.mle()
        self.cell_threshold = self.thresholds[0]-5

        # threshold the frame to generate the neuron mask
        self.frame.threshold(self.cell_threshold, x=255, y=0)
#
# end of file
