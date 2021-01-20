
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import imageio as iio
import imageproc as iproc

F_SRC = "~/neuropath/data/layer_segmentation/images/2013-223-03F_ANG_NeuN_1K_11-04-2020_RL.svs"
# F_SRC = "~/neuropath/data/layer_segmentation/images/2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs"

NAME = "~/neuropath/src/qupath-automatic-gm-segmentation/example_images/%s"
def main(argv):
    # retrieve the file
    if '-f' not in argv:
        f = F_SRC
    else:
        f = argv[argv.index('-f')+1]

    plot(NAME % "Tissue_Detection", f)

def plot(name, f):

    tissue_contours, thresholded, ts_detected, histo, ts_bg = \
        iproc.run_tissue_detection(f)

    # plotting...
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(histo, color='black')
    ax0.axvline(ts_bg, color='red')
    ax0.set_title('Grayscale Histogram of Slide')
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(thresholded, cmap='coolwarm')
    ax1.set_title('Thresholded Slide')
    ax2 = fig.add_subplot(gs[1,1])
    ax2.imshow(ts_detected)
    ax2.set_title('Tissue Detections')
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(iio.get_fullpath(name), dpi=500)

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
