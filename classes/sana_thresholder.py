
import os
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class Thresholder:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def run_gmm(self):
        self.gmm()
        self.mle()

    # generates the means and stdevs of the given data and pre-defined k value
    def gmm(self):
        gmm = GaussianMixture(
            n_components=self.k, covariance_type='full').fit(self.data)

        # get the sorted means and corresponding variances
        inds = gmm.means_[:, 0].argsort()
        means = gmm.means_[:, 0][inds]
        vars = gmm.covariances_[:, 0, 0][inds]

        self.means = np.rint(means).astype(int)
        self.vars = vars

    # finds the optimal boundaries between a set of means and vars
    def mle(self):

        # perform maximum likelihood estimation
        # NOTE: this is finding the crossing of the PDFs between the means
        t = np.arange(0, 256, 1)
        thresholds = []
        for i in range(self.k-1):

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
        for i in range(self.k-1):
            thresholds.append(self.means[i] + 2*np.sqrt(self.vars[i]))
        self.thresholds = thresholds
#
# end of Thresholder

class TissueThresholder(Thresholder):
    def __init__(self, frame, blur=5):
        self.frame = frame

        # convert to grayscale and blur
        self.frame.to_gray()
        self.frame.gauss_blur(blur)

        # flatten the data
        data = self.frame.img.flatten()[:, None]

        # initialize the thresholder
        super().__init__(data, 2)

    def mask_frame(self):

        # run the gmm algorithm to define the means and vars of the data
        self.gmm()

        # define threshold as 1 std from the mean
        self.close_right()
        self.tissue_threshold = self.thresholds[0]

        # threshold the frame to generate a tissue mask
        self.frame.threshold(self.tissue_threshold, x=1, y=0)
#
# end of class
