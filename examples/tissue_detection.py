
import os
import sys
import argparse
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sana_loader import Loader

SRC = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA = os.path.join(SRC, 'examples', 'data')

DEF_FILENAME = os.path.join(DATA, '2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs')

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

    # get the thumbnail from the SVS file
    frame = copy(loader.thumbnail)

    # perform the tissue mask
    frame.tissue_mask(blur=5)

    threshold = frame.tissue_threshold

    # plot the original thumbnail along with the tissue mask
    plot1(loader.thumbnail, frame)

def plot1(thumbnail, frame):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)

    # plot the color and grayscale histos
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(frame.color_histo[:, 0], color='red')
    ax0.plot(frame.color_histo[:, 1], color='green')
    ax0.plot(frame.color_histo[:, 2], color='blue')
    ax0.plot(frame.blur_histo, color='black')
    ax0.axvline(frame.tissue_threshold, linestyle='dashed', color='purple')
    ax0.set_title('Histogram of Slide')

    # plot the thumbnail
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(thumbnail.img)
    ax1.set_title('Slide Thumbnail')
    ax1.axis('off')
    ax1.grid('off')

    # plot the tissue mask
    ax2 = fig.add_subplot(gs[1,1])
    ax2.matshow(frame.img)
    ax2.set_title('Tissue Detection Mask')
    ax2.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.axis('off')
    ax2.grid('off')

    # finally, show the plot
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
