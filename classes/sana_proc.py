
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from scipy import signal

import sana_io
import sana_geo
from sana_loader import Loader
from sana_color_deconvolution import StainSeparator
from sana_framer import Framer
from sana_tiler import Tiler
from sana_frame import Frame

def estimate_stain_vector(est, stain, target):
    separator = StainSeparator(stain, target)
    separator.estimate_stain_vector(est)
    return separator.stain_vector

def calc_ao(frame, stain, target, slide_color,
            stain_vector=None, threshold=None, od=False):
    stain = separate_roi(frame, stain, target, stain_vector, od=od)

    # perform the thresholding
    thresholder = StainThresholder(stain.copy(), mx=slide_color)
    thresholder.mask_frame(threshold)
    thresh = thresholder.frame

    ao = np.mean(thresh.img)

    # if ao > 0.6:
    #     fig, axs = plt.subplots(1, 3)
    #     axs[0].imshow(stain.img)
    #     axs[1].imshow(thresh.img)
    #     axs[2].axvline(thresholder.stain_threshold, color='black')
    #     axs[2].axvline(thresholder.thresholds[-1], color='gray')
    #     axs[2].hist(stain.img.flatten(), bins=200)
    #     plt.show()


    return ao, thresholder.stain_threshold
#
# end of calc_ao

def roi_metrics():

    # calculate the ROI metrics
    #  GM_thickness - mean value in microns of 6 - 0
    #  GM_parallel - std value in microns of 6 - 0
    #  linearity_x - R^2 value of linear regression of layer x
    thickness = []
    for i in range(len(v_0)):
        thickness.append(v_6[i][1] - v_0[i][1])
    thickness = np.array(thickness, dtype=float)
    thickness *= loader.ds[args.level] / loader.ds[0]
    thickness *= loader.mpp
    gm_thickness = np.mean(thickness)
    gm_parallel = np.std(thickness)
    linearity_0 = sana_geo.linearity(x_0, y_0)
    linearity_6 = sana_geo.linearity(x_6, y_6)
    roi_metrics = [gm_thickness, gm_parallel, linearity_0, linearity_6]


    return roi_metrics
#
# end of segment_gm

#
# end of file
