
import os
import sys
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

# generates the means and stdevs of the given data and pre-defined k value
def gmm(data, k, n):
    gmm = GaussianMixture(
        n_components=k, covariance_type='full').fit(data)

    # get the sorted variances and corresponding means
    inds = gmm.covariances_[:, 0, 0].argsort()
    means = gmm.means_[:, 0][inds]
    vars = gmm.covariances_[:, 0, 0][inds]

    # take the top N distributions
    if data.dtype == np.uint8:
        means = np.rint(means[:n]).astype(np.uint8)
    else:
        means = means[:n]
    vars = vars[:n]

    # sort the distributions by the means
    inds = means.argsort()
    means = means[inds]
    vars = vars[inds]
    return means, vars

# finds the optimal boundaries between a set of means and vars
def mle(means, vars, mx=None):
    if means.dtype != np.uint8:
        if mx is None:
            mx = 1
        t = np.linspace(0, mx, 1000)
    else:
        t = np.arange(0, 256, 1)

    # perform maximum likelihood estimation
    # NOTE: this is finding the crossing of the PDFs between the means
    thresholds = np.zeros(len(means)-1)
    for i in range(len(means)-1):

        # generate the pdfs
        p1 = multivariate_normal(means[i], vars[i]).pdf(t)
        p2 = multivariate_normal(means[i+1], vars[i+1]).pdf(t)

        # only evaluate between the means
        a = np.where(t > means[i])[0][0]
        b = np.where(t < means[i+1])[0][-1]
        p1 = p1.flatten()[a:b]
        p2 = p2.flatten()[a:b]

        # find the last x value where p1 is more probable than p2
        inds = np.nonzero(p1 > p2)[0]
        if len(inds) == 0:
            thresholds[i] = -1
        else:
            thresholds[i] = t[inds[-1] + a]
    return thresholds

def knn(d, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    x = d.flatten()[:, None]
    _, labels, (centers) = cv2.kmeans(
        x.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    seg = centers[labels.flatten()].reshape(d.shape)
    centers = np.sort(centers[:, 0])

    thresholds = []
    for i in range(centers.shape[0]-1):
        x, y = centers[i], centers[i+1]
        thresholds.append((y-x)/2 + x)
    return seg, thresholds


def kittler(hist, mi=0, mx=255):

    # prepare the hist and bins
    h = hist.astype(np.float)
    g = np.arange(0, 257, 1).astype(np.float)

    # zero out data not within the min and max parameters
    h[:mi] = 0
    h[mx:] = 0

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
    if scaled:
        t = t * (mx - mi) / 255 + mi
    return [t]
#
# end of kittler
