
import os
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sana_color_deconvolution import StainSeparator
from matplotlib import pyplot as plt

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
        if self.data.dtype == np.uint8:
            means = np.rint(means[:self.n]).astype(int)
        else:
            means = means[:self.n]
        vars = vars[:self.n]

        # sort the distributionsn by the means
        inds = means.argsort()
        self.means = means[inds]
        self.vars = vars[inds]

    # finds the optimal boundaries between a set of means and vars
    def mle(self):
        if self.data.dtype != np.uint8:
            self.mle_float()
            return

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

    def mle_float(self):
        t = np.linspace(0, 1, 1000)
        thresholds = []
        for i in range(self.n-1):

            # generate the pdfs
            p1 = multivariate_normal(self.means[i], self.vars[i]).pdf(t)
            p2 = multivariate_normal(self.means[i+1], self.vars[i+1]).pdf(t)

            # only evaluate between the means
            a = np.where(t > self.means[i])[0][0]
            b = np.where(t < self.means[i+1])[0][-1]

            p1 = p1.flatten()[a:b]
            p2 = p2.flatten()[a:b]

            # find the last x value wehre p1 is more probable than p2
            try:
                thresholds.append(t[np.nonzero(p1 > p2)[0][-1] + a])
            except:
                if i == 1:
                    plt.hist(self.data, bins=256, range=(0,1), color='gray')
                    plt.plot(t, multivariate_normal(self.means[i], self.vars[i]).pdf(t), color='red')
                    plt.plot(t, multivariate_normal(self.means[i+1], self.vars[i+1]).pdf(t), color='blue')
                    plt.xlim((self.means[i], self.means[i+1]))
                    plt.show()
        self.thresholds = thresholds

    # defines the threshold as 1 std above the mean
    def close_right(self):
        thresholds = []
        for i in range(self.n-1):
            thresholds.append(self.means[i] + 2*np.sqrt(self.vars[i]))
        self.thresholds = thresholds

    def kittler(self, im):
        if im.dtype != np.uint8:
            mi, mx = np.min(im), np.max(im)
            im = 255 * (im - mi) / (mx - mi)
            im = im.astype(np.uint8)
        np.maximum(im, 1, out=im)
        h,g = np.histogram(im.ravel(),256,[0,256])
        h = h.astype(np.float)
        g = g.astype(np.float)
        g = g[:-1]
        c = np.cumsum(h)
        m = np.cumsum(h * g)
        s = np.cumsum(h * g**2)
        a = np.divide(s, c, out=np.zeros_like(s), where=c!=0)
        b = np.divide(m, c, out=np.zeros_like(m), where=c!=0)
        sigma_f = np.sqrt(a - b**2)
        cb = c[-1] - c
        mb = m[-1] - m
        sb = s[-1] - s
        ab = np.divide(sb, cb, out=np.zeros_like(s), where=cb!=0)
        bb = np.divide(mb, cb, out=np.zeros_like(mb), where=cb!=0)
        sigma_b = np.sqrt(ab - bb**2)
        p = 0 if c[-1] == 0 else c / c[-1]
        log_sigma_f = np.log(
            sigma_f, out=np.zeros_like(sigma_f), where=sigma_f!=0)
        log_sigma_b = np.log(
            sigma_b, out=np.zeros_like(sigma_b), where=sigma_b!=0)
        log_p = np.log(
            p, out=np.zeros_like(p), where=p!=0)
        log_1_p = np.log(
            1-p, out=np.zeros_like(p), where=p>1)
        v = p * log_sigma_f + (1-p)*log_sigma_b - p*log_p - (1-p)*log_1_p
        v[~np.isfinite(v)] = np.inf
        idx = np.argmin(v)
        t = g[idx]
        t = float(t) * ((mx - mi)/255) + mi
        self.thresholds = [t]
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
        self.mx = mx

        if self.frame.img.shape[-1] == 3:
            self.frame.to_gray()
            self.frame.gauss_blur(blur)

        # flatten the data and remove masked data from the analysis
        data = self.frame.img.flatten()
        data = data[data <= self.mx]
        data = data[:, None]

        # initalize the thresholder
        super().__init__(data, 3, 3)

    def mask_frame(self, thresholds=None):

        if thresholds is None:
            # run the gmm algorithm to define the means and vars of the data
            self.gmm()
            print(self.means, self.vars)

            # use mle to define the crossing of PDFs
            self.mle()
            self.stain_threshold = self.thresholds[-1]
            print(self.stain_threshold)

            # self.kittler(self.frame.img)
        else:
            self.thresholds = thresholds
            self.stain_threshold = thresholds[-1]

        if self.frame.img.dtype == np.uint8:
            self.frame.threshold(
                self.stain_threshold, x=1, y=0)
        else:
            self.frame.threshold(
                self.stain_threshold, x=0, y=1)
#
# end of StainThresholder

class CellThresholder(Thresholder):
    def __init__(self, frame, tissue_mask, blur=0, mi=0, mx=255):
        self.frame = frame
        self.mx = mx
        self.frame.to_gray()
        self.frame.gauss_blur(blur)

        # mask out any slide background
        self.frame.mask(tissue_mask, value=255)
        self.frame.mask_histo = self.frame.histogram()

        # flatten the data and remove masked data from the analysis
        data = self.frame.img[self.frame.img != 255].flatten()
        data = data[data <= self.mx]
        data = data[:, None]

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
