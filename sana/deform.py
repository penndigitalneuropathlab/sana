
import numpy as np
import cv2
from scipy.interpolate import interp1d, RegularGridInterpolator

import sana.geo
import sana.image

import torch
from torch.nn import functional as tfun

from matplotlib import pyplot as plt

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

def multi_fan_sample(lines, nh=None):
    
    # TODO: option for keeping distances equal
    nw = lines[0].shape[0]
    nsec = len(lines)-1
    if nh is None:
        nh = nsec*int(max(lines[-1][:,1]-lines[0][:,1])//nsec)
    sample_grid = np.zeros((nh, nw, 2))

    for i in range(lines.shape[1]):
        points = lines[:,i,:]
        x = np.zeros(nh)
        y = np.zeros(nh)
        for j, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
            x[int(j*nh//nsec):int((j+1)*nh//nsec)] = np.linspace(p0[0], p1[0], nh//nsec)
            y[int(j*nh//nsec):int((j+1)*nh//nsec)] = np.linspace(p0[1], p1[1], nh//nsec)

        sample_grid[:,i,:] = np.array((y,x)).T

    return sample_grid

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

def curved_fan_sample(w, h, top, bot, left, right, ax=None, ret_angle=False, plot_interval=250):
    # get the output dimensions of the image
    # NOTE: keeping h constant for now
    top_len = max(top[:,0]) - min(top[:,0])
    bot_len = max(bot[:,0]) - min(bot[:,0])
    nw = int(max(top_len, bot_len))
    nh = h

    # interpolate the boundaries
    top_interp = interp(top, nw)
    bot_interp = interp(bot, nw)

    # prepare arrays for the sampling grid
    sample_grid = np.zeros((nh, nw, 2))
    angles = np.zeros((nh, nw))

    l_a, l_b, l_c = np.polyfit(left[:,1], left[:,0], 2)
    r_a, r_b, r_c = np.polyfit(right[:,1], right[:,0], 2)
    a_sweep = np.linspace(l_a, r_a, nw)
    b_sweep = np.linspace(l_b, r_b, nw)
    c_sweep = np.linspace(l_c, r_c, nw)    
    
    # loop through the columns of the output image
    for i, (p0, p1, a, b, c) in enumerate(zip(top_interp, bot_interp, a_sweep, b_sweep, c_sweep)):
        y = np.linspace(p0[1], p1[1], h)
        x = a*y**2 + b*y + c
        xp = 2*a*y + b
        angle = 90 + np.rad2deg(np.arctan(xp))
            
        if i % plot_interval == 0 and not ax is None:
            ax.plot(x, y, color='red', linewidth=0.5)
    
        sample_grid[:,i,:] = np.array((y,x)).T
        angles[:,i] = angle
            
    return sample_grid, angles

def find_orthogonal_vectors(curve, y_intercept, x_intercept, ds=1):
    
    # interpolate the curve
    x = curve[:,0]
    y = curve[:,1]
    if not y_intercept is None:
        f = interp1d(x, y)
        st, en = np.min(curve[:,0]), np.max(curve[:,0])
        new_x = np.linspace(st, en, int(en-st), endpoint=True)
        new_y = f(new_x)
    else:
        f = interp1d(y, x)
        st, en = np.min(curve[:,1]), np.max(curve[:,1])
        new_y = np.linspace(st, en, int(en-st), endpoint=True)
        new_x = f(new_y)

    # get the orthoganal angle
    if not y_intercept is None:
        dy = np.gradient(new_y)
        dx = np.full_like(dy, new_x[1]-new_x[0])
    else:
        dx = np.gradient(new_x)
        dy = np.full_like(dx, new_y[1]-new_y[0])
    ths = np.arctan2(dy, dx) + np.pi/2

    # get the vector for each point on the curve
    vectors = []
    for i in range(0, new_x.shape[0], int(1/ds)):
        x0 = new_x[i]
        y0 = new_y[i]
        point_a = sana.geo.point_like(x0, y0, curve)

        th = ths[i]
        
        # find intersection point with the border
        m = np.tan(th)
        b = y0 - m * x0
        if not y_intercept is None:
            y1 = y_intercept
            x1 = (y1 - b) / m
        else:
            x1 = x_intercept
            y1 = m * x1 + b
        point_b = sana.geo.point_like(x1, y1, point_a)
        
        vectors.append(sana.geo.vector_like(*point_a, *point_b, curve))

    return vectors

    

def get_displacement_field_from_curves(a, b, c, d, w, h, debug=False):
    if a[0,0] < a[-1,1]:
        top_left = a[0]
        top_right = a[-1]
    else:
        top_left = a[-1]
        top_right = a[0]
    if b[0,0] < b[-1,1]:
        bottom_left = b[0]
        bottom_right = b[-1]
    else:
        bottom_left = b[-1]
        bottom_right = b[0]

    top_left = sana.geo.vector_like(*top_left, 0, 0, a)
    top_right = sana.geo.vector_like(*top_right, w, 0, a)
    bottom_left = sana.geo.vector_like(*bottom_left, 0, h, a)
    bottom_right = sana.geo.vector_like(*bottom_right, w, h, a)
    
    def plot_arrow(ax, v, color='blue'):
        head_size=10
        ax.arrow(
            v[0,0], v[0,1], v[1,0]-v[0,0], v[1,1]-v[0,1],
            color=color, 
            length_includes_head=True, 
            head_length=head_size,
            head_width=head_size
        )

    if debug:
        fig, ax = plt.subplots(1,1)
        [ax.plot(*x.T, color='black') for x in [a,b,c,d]]
        plot_arrow(ax, top_left)
        plot_arrow(ax, top_right)
        plot_arrow(ax, bottom_left)
        plot_arrow(ax, bottom_right)
        ax.invert_yaxis()
        
    vectors = []    
    ds = 0.05
    # TODO: orthogonal vectors should be clipped to image bounds
    vectors += find_orthogonal_vectors(a, 0, None, ds=ds)
    vectors += find_orthogonal_vectors(b, h, None, ds=ds)
    vectors += find_orthogonal_vectors(c, None, 0, ds=ds)
    vectors += find_orthogonal_vectors(d, None, w, ds=ds)
    # vectors += sweep_vectors_counterclockwise(a, w, h, axis=1, ds=ds)
    # vectors += sweep_vectors_counterclockwise(b, w, h, axis=1, ds=ds)
    # vectors += sweep_vectors_counterclockwise(c, w, h, axis=0, ds=ds)
    # vectors += sweep_vectors_counterclockwise(d, w, h, axis=0, ds=ds)
    border_vectors = np.array(vectors)

    # get the vectors w/ origin (0,0)
    # NOTE: inversing here
    vectors = border_vectors[:,0,:] - border_vectors[:,1,:]

    ds = 0.05
    ys = list(range(0, h, int(1/ds)))
    xs = list(range(0, w, int(1/ds)))
    displacement_field = np.zeros((len(ys), len(xs), 2, 2))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p0 = np.array((x,y))
        
            # get the weights of the vectors for this point based on the inverse distance from the vector origin
            # NOTE: using inverse of what we calculated above for grid_sample
            w = 1/np.sum(np.square(border_vectors[:,1,:] - p0), axis=1)
            w /= np.sum(w)
        
            # displacement = weighted average of the vectors 
            p1 = np.sum(w[:,None] * vectors, axis=0) + p0
            
            displacement_field[j,i,0] = p0
            displacement_field[j,i,1] = p1

    if debug:
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        [axs[0].plot(*x.T, color='black') for x in [a,b,c,d]]
        [plot_arrow(axs[0], x[::-1,:], color='blue') for x in border_vectors[::3]]
    
        [axs[1].plot(*x.T, color='black') for x in [a,b,c,d]]
        axs[0].invert_yaxis()

        plot_mod = 5
        for j in range(0, displacement_field.shape[0], plot_mod):
            for i in range(0, displacement_field.shape[1], plot_mod):
                plot_arrow(axs[1], displacement_field[j,i,:], color='red')

    return displacement_field

def get_vector_to_corner(a, w, h, axis):
    corners = np.array([
        [0,0],
        [w,0],
        [0,h],
        [w,h],
    ])

    # get the vector to the closest corner
    b = corners[np.argmin(np.sum(np.square(a-corners), axis=1))]
    v = sana.geo.vector_like(*a, *b, a)

    # handle tiny vectors, convert to small vertical or horizontal
    mag_thr = 0.01
    if v.get_length() < mag_thr:
        if axis == 1:
            if np.abs(v[1,0] - 0) < np.abs(v[1,0] - w):
                b = a.copy()
                b[1] -= mag_thr
            else:
                b = a.copy()
                b[1] += mag_thr
        else:
            if np.abs(v[1,1] - 0) < np.abs(v[1,1] - h):
                b = a.copy()
                b[0] -= mag_thr
            else:
                b = a.copy()
                b[0] += mag_thr
        v = sana.geo.vector_like(*a, *b, a)

    return v

def sweep_vectors_counterclockwise(curve, w, h, axis, ds=1):
    vectors = []

    # based on the axis, find the endpoints of the curve
    if axis == 1:
        a0 = curve[np.argmin(curve[:,0])]
        a1 = curve[np.argmax(curve[:,0])]
    else:
        a0 = curve[np.argmin(curve[:,1])]
        a1 = curve[np.argmax(curve[:,1])]
    
    # find the nearest corner to each endpoint
    v0 = get_vector_to_corner(a0, w, h, axis)
    v1 = get_vector_to_corner(a1, w, h, axis)    
    
    th0 = v0.get_angle()
    if th0 < 0:
        th0 += 2*np.pi
    th1 = v1.get_angle()
    if th1 < 0:
        th1 += 2*np.pi
        
    #print(180*th0/np.pi, 180*th1/np.pi, flush=True)
    
    # want to linspace from 300 -> 400 instead of 300 -> 40
    # TODO: what if opposite is true? might glitch out
    if (th0 >= 3*np.pi/2 and th0 < 2*np.pi and th1 > 0 and th1 < np.pi/2):
        th1 += 2*np.pi

    # angles are inversed, want to linspace in the positive direction (i.e counter clockwise)
    elif th0 > th1:
        temp = th0
        th0 = th1
        th1 = temp
        
        temp = v0
        v0 = v1
        v1 = temp

    #print(180*th0/np.pi, 180*th1/np.pi)
    
    # interpolate the curve
    x = curve[:,0]
    y = curve[:,1]
    if axis == 1:    
        N = np.abs(int(v1[0,0]-v0[0,0]))
        f = interp1d(x, y)
        new_x = np.linspace(v0[0,0], v1[0,0], N, endpoint=True)
        new_y = f(new_x)
    else:
        N = np.abs(int(v1[0,1]-v0[0,1]))
        f = interp1d(y, x)
        new_y = np.linspace(v0[0,1], v1[0,1], N, endpoint=True)
        new_x = f(new_y)

    # rotate 1st angle counter clockwise to 2nd angle
    ths = np.linspace(th0, th1, N, endpoint=True)

    # TODO: if len(v) == 0, then we need to define it to be up/down/left/right based on the seg

    # get the vector for each point on the curve
    for i in range(0, N, int(1/ds)):
        x0 = new_x[i]
        y0 = new_y[i]
        point_a = sana.geo.point_like(x0, y0, v0)
        
        th = ths[i]

        # find intersection point with the border
        m = np.tan(th)
        b = y0 - m * x0
        if axis == 1:
            y1 = v0[1,1]
            x1 = (y1 - b) / m
        else:
            x1 = v0[1,0]
            y1 = m * x1 + b
        point_b = sana.geo.point_like(x1, y1, point_a)
        
        vectors.append(sana.geo.vector_like(*point_a, *point_b, v0))

    return vectors

def get_homography(a, b, c, d, w, h, debug=False):
    if a[0,0] < a[-1,1]:
        top_left = a[0]
        top_right = a[-1]
    else:
        top_left = a[-1]
        top_right = a[0]
    if b[0,0] < b[-1,1]:
        bottom_left = b[0]
        bottom_right = b[-1]
    else:
        bottom_left = b[-1]
        bottom_right = b[0]

    top_left = sana.geo.vector_like(*top_left, 0, 0, a)
    top_right = sana.geo.vector_like(*top_right, w, 0, a)
    bottom_left = sana.geo.vector_like(*bottom_left, 0, h, a)
    bottom_right = sana.geo.vector_like(*bottom_right, w, h, a)

    def plot_arrow(ax, v, color='blue'):
        head_size=10
        ax.arrow(
            v[0,0], v[0,1], v[1,0]-v[0,0], v[1,1]-v[0,1],
            color=color, 
            length_includes_head=True, 
            head_length=head_size,
            head_width=head_size
        )
    
    vectors = []    
    ds = 0.05
    vectors += sweep_vectors_counterclockwise(a, w, h, axis=1, ds=ds)
    vectors += sweep_vectors_counterclockwise(b, w, h, axis=1, ds=ds)
    vectors += sweep_vectors_counterclockwise(c, w, h, axis=0, ds=ds)
    vectors += sweep_vectors_counterclockwise(d, w, h, axis=0, ds=ds)
    border_vectors = np.array(vectors)


    if debug:
        fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
        [axs.plot(*x.T, color='black') for x in [a,b,c,d]]
        [plot_arrow(axs, x[::-1,:], color='blue') for x in border_vectors[::3]]
        axs.invert_yaxis()

    src = border_vectors[:,0,:]
    dst = border_vectors[:,1,:]
    src = np.array([
        top_left[0,:],
        top_right[0,:],
        bottom_left[0,:],
        bottom_right[0,:],
    ])
    dst = np.array([
        top_left[1,:],
        top_right[1,:],
        bottom_left[1,:],
        bottom_right[1,:],
    ])
    H, mask = cv2.findHomography(src, dst)

    return H

def apply_displacement(frame, displacement_field):

    # interpolate the displacement field
    grid_frame = sana.image.frame_like(frame, displacement_field[:,:,1,:].astype(float))
    grid_frame.resize(frame.size(), interpolation=cv2.INTER_LINEAR)
    
    # prepare the tensors
    src = frame.img.transpose(2,0,1)
    src = src[None,...].astype(float)
    grid = grid_frame.img[None,...]

    # normalize points from -1 to 1
    grid[...,0] = (2*grid[...,0] / frame.size()[0]) - 1
    grid[...,1] = (2*grid[...,1] / frame.size()[1]) - 1

    src_tensor = torch.from_numpy(src)
    grid_tensor = torch.from_numpy(grid)

    res = tfun.grid_sample(src_tensor, grid_tensor, align_corners=False)
    res = res.numpy()[0].transpose(1,2,0)
    res = sana.image.frame_like(frame, res)
    if res.is_rgb():
        res.to_short()
    #res.resize(frame.size(), interpolation=cv2.INTER_LINEAR)
    
    return res
