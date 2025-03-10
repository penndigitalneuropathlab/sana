
import numpy as np
import cv2
from scipy.interpolate import interp1d, RegularGridInterpolator

import sana.geo
import sana.image

import torch
from torch.nn import functional as tfun

from matplotlib import pyplot as plt

def fit_parabola(c, N):
    c = interp(c, N)
    x, y = c[:,0], c[:,1]
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    xnew = np.linspace(np.min(x), np.max(x), N)
    ynew = p(xnew)
    return sana.geo.curve_like(xnew, ynew, c)

def interp(c, N, xmi=None, xmx=None):
    x = c[:,0]
    y = c[:,1]
    f = interp1d(x, y)
    if xmi is None:
        xmi = x[0]
    if xmx is None:
        xmx = x[-1]
    xp = np.linspace(xmi, xmx, N)
    yp = f(xp)
    return sana.geo.curve_like(xp, yp, c)

def separate_curve_at_point(a, xp, yp):

    # get the vertices that the point is inbetween
    mi_dist = np.inf
    for i in range(a.shape[0]-1):
        x0, y0 = a[i]
        x1, y1 = a[i+1]
        m = (y1-y0)/(x1-x0)
        b = y0 - x0*m
        dist = np.abs(xp*m + b - yp)
        if dist < mi_dist:
            mi_dist = dist
            idx = i
    return idx
            
def clip_curve(a, x0, y0, x1, y1):
    """
    this function clips the given curve to the points p0 and p1
    """
    idx0 = separate_curve_at_point(a, x0, y0)
    idx1 = separate_curve_at_point(a, x1, y1)    
    if idx1 < idx0:
        idx0, idx1 = idx1, idx0
        x0, y0, x1, y1 = x1, y1, x0, y0
    a = a[idx0:idx1+2]
    a[0] = (x0, y0)
    a[-1] = (x1, y1)

    return a

def intersect_curves(a, b, N=1000):
    """
    this function finds the interpolated point that exists in both input curves
    """
    # interpolate between vertices in the curves
    x_a, y_a = interp(a, N).T
    x_b, y_b = interp(b, N).T

    # get the distance of each point in a to each point in b (i.e. NxN matrix)
    dist = (x_a[:,None]-x_b)**2 + (y_a[:,None]-y_b)**2
    
    # get the point in a closest to a point in b
    idx = np.unravel_index(np.argmin(dist), (N,N))[0]
    x_int, y_int = x_a[idx], y_a[idx]

    return x_int, y_int

def clip_between_segments(a, b, c):
    """
    this function clips curve "a" at the intersection points with curves "b" and "c"
    """
    abx, aby = intersect_curves(a, b)
    acx, acy = intersect_curves(a, c)
    return clip_curve(a, acx, acy, abx, aby)        

def grid_sample(frame, sample_grid):

    nchan = frame.img.shape[2]

    h, w = frame.img.shape[:2]
    nh, nw = sample_grid.shape[:2]
    out = np.zeros((nh, nw, nchan))
    for channel in range(out.shape[2]):

        # define the grid interpolator on the current channel
        x = np.arange(0, w)
        y = np.arange(0, h)
        grid_interp = RegularGridInterpolator(
            (y, x), 
            frame.img[:,:,channel], 
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        out[:,:,channel] = grid_interp(sample_grid)

    out_frame = sana.image.Frame(out)
    if frame.is_short():
        out_frame.to_short()

    return out_frame

def fan_sample(top, bot, ax=None):

    # new width is the longest given segmentation
    top_len = max(top[:,0]) - min(top[:,0])
    bot_len = max(bot[:,0]) - min(bot[:,0])
    nw = int(max(top_len, bot_len))

    # new height is the largest height between segmentations
    nh = int(max(bot[:,1]-top[:,1]))

    # interpolate the boundaries
    top_interp = interp(top, nw)
    bot_interp = interp(bot, nw)

    # prepare arrays for the sampling grid
    sample_grid = np.zeros((nh, nw, 2))
    angles = np.zeros((nh, nw))

    # loop through the columns of the output image
    for i, (p0, p1) in enumerate(zip(top_interp, bot_interp)):

        # define the sampling curve for this column by drawing a linear equation
        x = np.linspace(p0[0], p1[0], nh)
        y = np.linspace(p0[1], p1[1], nh)

        # calculate angle from the slope of the linear equation
        angle = np.rad2deg(-np.arctan((y[1]-y[0])/(x[1]-x[0])))

        # debugging line plots
        if i % 250 == 0 and not ax is None:
            ax.plot(x, y, color='red', linewidth=0.5)

        sample_grid[:,i,:] = np.array((y,x)).T
        angles[:,i] = angle

    angles = np.where(angles < 0, angles+180, angles)

    return sample_grid, angles
    
# TODO: sometimes the weird ROIs don't deform properly need a way to clip the sampling curves to the top/bot curves
def curved_fan_sample(top, bot, left, right, ax=None, ret_angle=False, plot_interval=250):

    # rotate so the ROI is orthogonal
    angle = top.get_angle()
    ctrx = np.min([np.min(x[:,0]) for x in [top, bot, left, right]]) + \
        np.max([np.max(x[:,0]) for x in [top, bot, left, right]])
    ctry = np.min([np.min(x[:,1]) for x in [top, bot, left, right]]) + \
        np.max([np.max(x[:,1]) for x in [top, bot, left, right]])
    p = sana.geo.point_like(ctrx, ctry, top)
    top.rotate(p, -angle)
    bot.rotate(p, -angle)
    left.rotate(p, -angle)
    right.rotate(p, -angle)
    if np.mean(top[:,1]) > np.mean(bot[:,1]):
        top.rotate(p, 180)
        bot.rotate(p, 180)
        left.rotate(p, 180)
        right.rotate(p, 180)
        angle += 180
        
    # set output dimensions to stretch boundaries rather than compress
    nw = int(max(np.abs(max(top[:,0]) - min(top[:,0])), np.abs(max(bot[:,0]) - min(bot[:,0]))))
    nh = int(max(np.abs(max(left[:,1]) - min(left[:,1])), np.abs(max(right[:,1]) - min(right[:,1]))))

    # fit parabolas to the boundaries
    top = fit_parabola(top, nw)
    bot = fit_parabola(bot, nw)
    left = fit_parabola(left[:,::-1], nh)[:,::-1]
    right = fit_parabola(right[:,::-1], nh)[:,::-1]    
    if not ax is None:
        [x.rotate(p, angle) for x in [top, bot, left, right]]
        ax.plot(*top.T, color='black', linewidth=0.5)
        ax.plot(*bot.T, color='black', linewidth=0.5)
        ax.plot(*left.T, color='black', linewidth=0.5)
        ax.plot(*right.T, color='black', linewidth=0.5)
        [x.rotate(p, -angle) for x in [top, bot, left, right]]        
        
    # prepare arrays for the sampling grid
    sample_grid = np.zeros((nh, nw, 2))
    angles = np.zeros((nh, nw))

    # get the parabola coefficients of the left and right edges
    l_a, l_b, l_c = np.polyfit(left[:,1], left[:,0], 2)
    r_a, r_b, r_c = np.polyfit(right[:,1], right[:,0], 2)

    # sweep the coefficients to morph L into R
    a_sweep = np.linspace(l_a, r_a, nw)
    b_sweep = np.linspace(l_b, r_b, nw)
    c_sweep = np.linspace(l_c, r_c, nw)    
    
    # loop through the columns of the output image
    for i, (p0, p1, a, b, c) in enumerate(zip(top, bot, a_sweep, b_sweep, c_sweep)):

        # define the input space for this column
        y = np.linspace(p0[1], p1[1], nh)

        # calculate the output space for the column
        x = a*y**2 + b*y + c

        # differentiate to find the angle of rotation at each output pixel
        xp = 2*a*y + b
        sampling_angle = np.rad2deg(np.arctan(xp))
        
        # rotate the sampling curve back to the original coordinate system
        sampling_curve = sana.geo.curve_like(x, y, top)
        sampling_curve.rotate(p, angle)
        if i % plot_interval == 0 and not ax is None:
            ax.plot(*sampling_curve.T, color='red', linewidth=0.5)

        # store the sampling curve for this output column
        sample_grid[:,i,:] = sampling_curve[:,::-1]

        # store the amount of clockwise degrees each pixel is being rotated
        angles[:,i] = sampling_angle

    # include the amount of rotation being applied by orthogonalizing the top
    angles -= angle

    angles = angles % 360
    angles = (angles + 360) % 360
    angles = np.where(angles <= 180, angles, angles - 360)
        
    return sample_grid, angles
