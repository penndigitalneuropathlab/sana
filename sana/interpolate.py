
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import scipy.optimize
from numba import jit

import sana.geo
import sana.image

def fit_polynomial(c, degrees, N, fixed_endpoints=True, x0=None, x1=None):
    c = interp(c, N)
    x, y = c.T
    sigma = np.ones_like(x)
    if fixed_endpoints:
        sigma[[0, -1]] = 0.01
    def f(x, *z):
        return np.poly1d(z)(x)
    z, _ = scipy.optimize.curve_fit(f, x, y, p0=(0,)*(degrees+1), sigma=sigma)
    p = np.poly1d(z)
    if x0 is None:
        x0 = np.min(x)
    if x1 is None:
        x1 = np.max(x)
    xnew = np.linspace(x0, x1, N)
    ynew = p(xnew)
    return sana.geo.curve_like(c, xnew, ynew), z

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
    return sana.geo.curve_like(c, xp, yp)

def separate_curve_at_point(a, xp, yp):

    # get the vertices that the point is inbetween
    mi_dist = np.inf
    for i in range(a.shape[0]-1):
        x0, y0 = a[i]
        x1, y1 = a[i+1]
        den = x1-x0
        if den == 0:
            dist = np.abs(xp - x0)
        else:
            m = (y1-y0)/den
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
    a = a[idx0:idx1+2].copy()
    a[0] = (x0, y0)
    a[-1] = (x1, y1)

    return a

@jit(nopython=True)
def intersect_curves(c1: np.ndarray, c2: np.ndarray, rtol: float=1e-3):
    """
    this function finds the interpolated point that exists in both input curves
    """
    
    # calculate linear equation of each segment in the 1st curve
    for i in range(c1.shape[0]-1):
        x1, y1 = c1[i:i+2].T
        den = x1[1]-x1[0]
        if den == 0:
            m1 = np.inf
        else:
            m1 = (y1[1]-y1[0]) / den
        b1 = c1[i,1] - m1*c1[i,0]

        # do the same for the 2nd curve
        for j in range(c2.shape[0]-1):
            x2, y2 = c2[j:j+2].T
            den = x2[1]-x2[0]
            if den == 0:
                m2 = np.inf
            else:
                m2 = (y2[1]-y2[0]) / den
            b2 = c2[j,1] - m2*c2[j,0]

            # get the intersection between the 2 linear eqs
            if m1 == np.inf and m2 != np.inf:
                xp = x2[0]
                yp = m2 * xp + b2
            elif m2 == np.inf and m1 != np.inf:
                xp = x1[0]
                yp = m1 * xp + b1  
            elif m1 != np.inf and m2 != np.inf and (m1-m2) != 0:
                xp = -(b1 - b2) / (m1 - m2)
                yp = m1 * xp + b1
                if ((np.min(x1) < xp) or np.isclose(np.min(x1), xp, rtol=rtol)) and \
                   ((xp < np.max(x1)) or np.isclose(xp, np.max(x1), rtol=rtol)) and \
                   ((np.min(x2) < xp) or np.isclose(np.min(x2), xp, rtol=rtol)) and \
                   ((xp < np.max(x2)) or np.isclose(xp, np.max(x2), rtol=rtol)) and \
                   ((np.min(y1) < yp) or np.isclose(np.min(y1), yp, rtol=rtol)) and \
                   ((yp < np.max(y1)) or np.isclose(yp, np.max(y1), rtol=rtol)) and \
                   ((np.min(y2) < yp) or np.isclose(np.min(y2), yp, rtol=rtol)) and \
                   ((yp < np.max(y2)) or np.isclose(yp, np.max(y2), rtol=rtol)):
                    return xp, yp

def clip_between_segments(a, b, c):
    """
    this function clips curve "a" at the intersection points with curves "b" and "c"
    """
    abx, aby = intersect_curves(np.array(a), np.array(b))
    acx, acy = intersect_curves(np.array(a), np.array(c))

    return clip_curve(a, acx, acy, abx, aby)        

def clip_quadrilateral_segments(top, right, bottom, left):
    new_top = clip_between_segments(top, right, left)
    new_right = clip_between_segments(right, bottom, top)
    new_bottom = clip_between_segments(bottom, left, right)
    new_left = clip_between_segments(left, top, bottom)
    return new_top, new_right, new_bottom, new_left

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

    out_frame = sana.image.Frame(out, is_deformed=True)
    if frame.is_short():
        out_frame.to_short()
        
    return out_frame

def fan_sample(top, right, bottom, left, degrees=1, N=10, ax=None, plot_interval=250):
    top = top.copy()
    right = right.copy()
    bottom = bottom.copy()
    left = left.copy()
    
    # rotate so the ROI is orthogonal
    angle = top.get_angle()
    ctrx = np.min([np.min(x[:,0]) for x in [top, right, bottom, left]]) + \
        np.max([np.max(x[:,0]) for x in [top, right, bottom, left]])
    ctry = np.min([np.min(x[:,1]) for x in [top, right, bottom, left]]) + \
        np.max([np.max(x[:,1]) for x in [top, right, bottom, left]])
    ctr = sana.geo.point_like(top, ctrx, ctry)
    [x.rotate(ctr, -angle) for x in [top, right, bottom, left]]
    if np.mean(top[:,1]) > np.mean(bottom[:,1]):
        [x.rotate(ctr, 180) for x in [top, right, bottom, left]]
        angle += 180
        
    # set output dimensions to stretch the image to the boundaries of the segmentations
    xbound = max([np.max(curve[:,0]) for curve in [top, right, bottom, left]])
    ybound = max([np.max(curve[:,1]) for curve in [top, right, bottom, left]])
    top_width = max(top[:,0]) - min(top[:,0])
    bottom_width = max(bottom[:,0]) - min(bottom[:,0])
    nw = int(max(top_width, bottom_width))
    left_height = max(left[:,1]) - min(left[:,1])
    right_height = max(right[:,1]) - min(right[:,1])
    nh = int(max(left_height, right_height))

    # fit polynomials to the boundaries
    right, right_z = fit_polynomial(right[:,::-1], degrees, nh)
    right = right[:,::-1]
    left, left_z = fit_polynomial(left[:,::-1], degrees, nh)
    left = left[:,::-1]
    top, _ = fit_polynomial(top, degrees, N, x0=0, x1=xbound)
    bottom, _ = fit_polynomial(bottom, degrees, N, x0=0, x1=xbound)

    if not ax is None:
        [x.rotate(ctr, angle) for x in [top, right, bottom, left]]
        ax.plot(*top.T, color='black', linewidth=2)
        ax.plot(*right.T, color='black', linewidth=2)        
        ax.plot(*bottom.T, color='black', linewidth=2)
        ax.plot(*left.T, color='black', linewidth=2)
        [x.rotate(ctr, -angle) for x in [top, right, bottom, left]]        
    # prepare arrays for the sampling grid
    sample_grid = np.zeros((nh, nw, 2))
    angles = np.zeros((nh, nw))

    # sweep the polynomial coefficients to morph the left boundary into the right
    z_sweep = np.zeros((nw, degrees+1))
    for i in range(degrees+1):
        z_sweep[:,i] = np.linspace(left_z[i], right_z[i], nw)

    # loop through the columns of the output image
    for i in range(nw):

        # get the polynomial for this column
        z = z_sweep[i]

        # define the input space for this column
        y = np.linspace(-100, nh+100, N)

        # calculate the output space for the column
        p = np.poly1d(z)
        x = p(y)
        sampling_curve = sana.geo.curve_like(top, x, y)
        
        # clip the curve to the top/bottom curves
        sampling_curve = clip_between_segments(sampling_curve, top, bottom)
        sampling_curve = interp(sampling_curve[:,::-1], nh)[:,::-1]

        # differentiate to find the angle of rotation at each output pixel
        pp = np.polyder(p)
        sampling_angle = np.rad2deg(np.arctan(pp(sampling_curve[:,1])))
        
        # rotate the sampling curve back to the original coordinate system
        sampling_curve.rotate(ctr, angle)
        if i % plot_interval == 0 and not ax is None:
            ax.plot(*sampling_curve.T, color='red', linewidth=0.5)

        # store the sampling curve for this output column
        sample_grid[:,i,:] = sampling_curve[:,::-1]

        # store the amount of clockwise degrees each pixel is being rotated
        angles[:,i] = sampling_angle

    # include the amount of rotation being applied by orthogonalizing the top
    angles -= angle

    # limit the range of angle values
    angles = angles % 360
    angles = (angles + 360) % 360
    angles = np.where(angles <= 180, angles, angles - 360)
        
    return sample_grid, angles
