
import os
import sys
import argparse
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sana_loader import Loader
from sana_thresholder import TissueThresholder
from sana_detector import TissueDetector

SRC = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA = os.path.join(SRC, 'examples', 'data')

DEF_FILENAME = os.path.join(DATA, 'images', '2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs')

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str)
    args = parser.parse_args()

    if args.filename is None:
        filename = DEF_FILENAME
    else:
        filename = args.filename

    # initialize the Loader
    loader = Loader(filename)
    loader.set_lvl(loader.lc-1)

    # get the thumbnail from the SVS file
    frame = copy(loader.thumbnail)

    # threshold the frame and generate the tissue mask
    thresholder = TissueThresholder(frame, blur=5)
    thresholder.mask_frame()
    tissue_threshold = thresholder.tissue_threshold

    # perform object detection on the tissue mask
    detector = TissueDetector(loader.mpp, loader.ds, loader.lvl)
    detector.run(frame, min_body_area=1e7, min_hole_area=1e6)

    # get the tissue body detections
    tissue_detections = detector.get_bodies()

    # get just the polygons and convert back to pixel resolution
    polygons = [d.polygon for d in tissue_detections]
    [p.to_pixels(loader.lvl) for p in polygons]

    # plot the original thumbnail along with the tissue mask, and the objects
    plot1(loader.thumbnail, frame, tissue_threshold, polygons)

    # finally, show the plot
    plt.show()

def plot1(thumbnail, frame, threshold, polygons):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,3)

    # plot the color and grayscale histos
    ax0 = fig.add_subplot(gs[0,:])
    ax0.bar(list(range(0, 256)), frame.blur_histo[:, 0], color='black', label='Intensity Histogram')
    ax0.plot(frame.color_histo[:, 0], color='red')
    ax0.plot(frame.color_histo[:, 1], color='green')
    ax0.plot(frame.color_histo[:, 2], color='blue')
    ax0.axvline(threshold, linestyle='dashed', color='purple', label='Tissue/Slide Threshold')
    ax0.set_xlim([180, 255])
    ax0.get_yaxis().set_visible(False)
    ax0.legend()

    # plot the thumbnail
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(thumbnail.img)
    ax1.axis('off')
    ax1.grid('off')

    # plot the tissue mask
    ax2 = fig.add_subplot(gs[1,1])
    ax2.matshow(1 - frame.img)
    ax2.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.axis('off')
    ax2.grid('off')

    # plot the tissue objects
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(thumbnail.img)
    ax3.axis('off')
    ax3.grid('off')
    for p in polygons:
        ax3.plot(p.x, p.y, color='black')

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
