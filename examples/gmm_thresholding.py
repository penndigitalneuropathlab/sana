
import os
import sys
import argparse
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_proc
from sana_loader import Loader
from sana_thresholder import StainThresholder

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
    frame = loader.thumbnail.copy()
    frame = sana_proc.separate_roi(frame, 'H-DAB', 'DAB')

    # threshold the frame and generate the tissue mask
    thresholder = StainThresholder(frame, blur=5, mi=180, mx=255)
    thresholder.mask_frame()

    # plot the original thumbnail along with the tissue mask, and the objects
    plot1(loader.thumbnail, frame, thresholder)

    # finally, show the plot
    plt.show()

def plot1(thumbnail, frame, thresholder):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,:])

    # plot the gmm approximation
    sm, tm, bm = thresholder.means
    sv, tv, bv = thresholder.vars
    stain_threshold = int(thresholder.stain_threshold)
    tissue_threshold = int(thresholder.tissue_threshold)
    x = np.linspace(0, 256, 1000)
    # ax0.plot(x, stats.norm.pdf(x, sm, sv), color='brown', label='Stain Distribution')
    # ax0.plot(x, stats.norm.pdf(x, tm, tv), color='purple', label='Tissue Distribution')
    # ax0.plot(x, stats.norm.pdf(x, bm, bv), color='gray', label='Slide Distribution')
    ax0.axvline(stain_threshold, linestyle='dashed', color='black')
    ax0.axvline(tissue_threshold, linestyle='dashed', color='black')

    # plot the separated histograms
    hist = frame.blur_histo / (frame.size[0] * frame.size[1])
    shist = hist[:stain_threshold, 0]
    thist = hist[stain_threshold:tissue_threshold, 0]
    bhist = hist[tissue_threshold:, 0]
    ax0.bar(np.arange(0, stain_threshold), shist,
            color='brown', label='Stain Histogram')
    ax0.bar(np.arange(stain_threshold, tissue_threshold), thist,
            color='purple', label='Tissue Histogram')
    ax0.bar(np.arange(tissue_threshold, 256), bhist,
            color='gray', label='Slide Histogram')
    ax0.set_xlim([180, 255])
    ax0.get_yaxis().set_visible(False)
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(thumbnail.img)
    ax1.axis('off')
    ax1.grid('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.matshow(frame.img)
    ax2.axis('off')
    ax2.grid('off')

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
