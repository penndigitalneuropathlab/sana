
import os
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
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
def mle(means, vars):
    if means.dtype != np.uint8:
        t = np.linspace(0, 1, 1000)
    else:
        t = np.arange(0, 256, 1)

    # perform maximum likelihood estimation
    # NOTE: this is finding the crossing of the PDFs between the means
    thresholds = []
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
        thresholds.append(np.nonzero(p1 > p2)[0][-1] + a)
    return thresholds

def kittler(data):
    if data.dtype != np.uint8:
        scaled = True
        mi, mx = np.min(data), np.max(data)
        data = 255 * (data - mi) / (mx - mi)
        data = np.rint(data).astype(np.uint8)
    else:
        scaled = False
    np.maximum(data, 1, out=data)
    h,g = np.histogram(data,256,[0,256])
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
    if scaled:
        t = t * (mx - mi) / 255 + mi
    return [t]
