
import os
import sys
import argparse
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import sana_geo
from sana_loader import Loader
from sana_framer import Framer
from sana_thresholder import TissueThresholder, NeuronThresholder
from sana_detector import TissueDetector#, NeuronDetector

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
    loader.set_lvl(0)

    # initialize the framer for the neuron ROIs
    fsize = sana_geo.Point(1500, 1500, loader.mpp, loader.ds)
    flocs = [sana_geo.Point(9500, 10500, loader.mpp, loader.ds)]
    framer = Framer(loader, fsize, locs=flocs)

    # create the tissue mask
    tissue_frame = copy(loader.thumbnail)
    thresholder = TissueThresholder(tissue_frame, blur=5)
    thresholder.mask_frame()
    detector = TissueDetector(loader.mpp, loader.ds, loader.lc-1)
    detector.run(tissue_frame, min_body_area=1e7, min_hole_area=1e6)
    tissue_mask = detector.generate_mask(tissue_frame.size)

    for i in range(framer.n):

        # load the frame
        frame = framer.load(i)
        neuron_mask = copy(frame)

        # TODO: this really needs cleaned up
        # crop and rescale the tissue mask into the proper location
        crop_loc = copy(framer.locs[i])
        crop_size = copy(framer.size)
        sana_geo.rescale(crop_loc, loader.lc-1)
        sana_geo.rescale(crop_size, loader.lc-1)
        tissue_mask = tissue_mask.crop(crop_loc, crop_size)
        ds = int(round(loader.ds[loader.lc-1] / loader.ds[loader.lvl]))
        tissue_mask = tissue_mask.rescale(ds, size=neuron_mask.size)

        # threshold the frame and generate the neuron mask
        thresholder = NeuronThresholder(neuron_mask, tissue_mask)
        thresholder.mask_frame()
        neuron_threshold = thresholder.neuron_threshold

        # # perform object detection on the tissue mask
        # detector = TissueDetector(loader)
        # detector.run(frame, min_body_area=1e7, min_hole_area=1e6)
        #
        # # get the tissue body detections
        # tissue_detections = detector.get_bodies()
        #
        # # get just the polygons and convert back to pixel resolution
        # polygons = [d.polygon for d in tissue_detections]
        # [p.to_pixels(loader.lvl) for p in polygons]

        # plot the original roi with the neuron mask
        plot1(frame, neuron_mask, neuron_threshold)

        # plot the thresholded thumbnail along with the tissue detections
        # plot2(loader.thumbnail, frame, polygons)

        # finally, show the plot
        plt.show()

def plot1(frame, mask, threshold):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)

    # plot the color and grayscale histos
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(frame.color_histo[:, 0], color='red')
    ax0.plot(frame.color_histo[:, 1], color='green')
    ax0.plot(frame.color_histo[:, 2], color='blue')
    ax0.plot(mask.mask_histo[:255, 0], color='black')
    ax0.axvline(threshold, linestyle='dashed', color='purple')
    ax0.set_title('Histogram of Slide')

    # plot the thumbnail
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(frame.img)
    ax1.set_title('Original Slide')
    ax1.axis('off')
    ax1.grid('off')

    # plot the tissue mask
    ax2 = fig.add_subplot(gs[1,1])
    ax2.matshow(mask.img)
    ax2.set_title('Neuron Detection Mask')
    ax2.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.axis('off')
    ax2.grid('off')

def plot2(thumbnail, frame, polygons):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.matshow(frame.img)
    ax0.set_title('Tissue Detection Mask')
    ax0.axis('off')
    ax0.grid('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(thumbnail.img)
    ax1.set_title('Tissue Object Detections')
    ax1.axis('off')
    ax1.grid('off')
    for p in polygons:
        ax1.plot(p.x, p.y, color='black')

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
