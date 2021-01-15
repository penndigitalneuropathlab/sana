
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import imageio as iio
import imageproc as iproc

F_SRC = "~/neuropath/data/layer_segmentation/images/2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs"

NAME = "~/neuropath/src/qupath-automatic-gm-segmentation/example_images/%s"
def main(argv):
    # retrieve the file
    if '-f' not in argv:
        f = F_SRC
    else:
        f = argv[argv.index('-f')+1]

    lvl = 2
    plot(NAME % "Tissue_Detection_Full", f, lvl)

    lvl = 1
    loc = np.array((14000,9500))
    size = np.array((1000, 800))
    plot(NAME % "Tissue_Detection_Zoomed", f, lvl, loc, size)

def plot(name, f, lvl, loc=None, size=None):

    loader = iio.SVSLoader(f)

    loader.set_lvl(lvl)

    if loc is None:
        img = loader.load_thumbnail()
    else:
        loc = loader.micron_to_px(loc)
        size = loader.micron_to_px(size)
        img = loader.load_region(loc, size)

    histo = iproc.histogram(iproc.round_img(iproc.rgb_to_gray(img)))
    tissue_contours, thresholded, tissue_detected = iproc.detect_tissue(
        loader, img, lvl)

    # plotting...
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(histo, color='black')
    ax0.axvline(237, color='red', label='Tissue/Slide Background Threshold')
    ax0.set_title('Grayscale Histogram of Slide')
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(thresholded, cmap='coolwarm')
    ax1.set_title('Thresholded Slide')
    ax2 = fig.add_subplot(gs[1,1])
    ax2.imshow(tissue_detected)
    ax2.set_title('Tissue Detections')
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(iio.get_fullpath(name))

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
