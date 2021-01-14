
import os
import sys
import cv2
import struct
import numpy as np
from matplotlib import pyplot as plt
import imageio as iio
import imageproc as iproc

F_SRC = "~/neuropath/data/layer_segmentation/images/2011-024-37F_STC_NeuN_1K_11-04-2020_RL.svs"
D_SRC = 'tissue_detections'

# TODO: profile this code to optimize
def main(argv):

    # retrieve the file
    if '-f' not in argv:
        f = F_SRC
    else:
        f = argv[argv.index('-f')+1]
    if '-d' not in argv:
        d = D_SRC
    else:
        d = argv[argv.index('-d')+1]

    # initialize the stain separation object
    separator = iproc.StainSeparator('H-DAB')

    # load the tissue specimen detections in microns
    tissue_detections = iio.read_tissue_detections(f, d)

    # initialize the loader
    loader = iio.SVSLoader(f)

    # set the level used for processing
    lvl = 2
    loader.set_lvl(lvl)

    # load the thumbnail image for plotting later on
    thumbnail = loader.load_thumbnail().astype(np.int32)

    # get the color of the slide background
    slide_color = iproc.get_slide_color(thumbnail)

    # define the frame size in relative pixel resolution
    frame_size = np.array((2000,2000))
    frame_step = np.array((2000,2000))

    # define the tile size in relative pixel resolution
    tile_size_m = np.array((100, 100))
    tile_step_m = np.array((50, 50))
    tile_size = loader.micron_to_px(tile_size_m)
    tile_step = loader.micron_to_px(tile_step_m)

    # pad the frame size by the tile size
    frame_pad = tile_size - 1
    frame_size += frame_pad

    # get the number of tiles in the frame
    ntiles = loader.get_ntiles(tile_size, tile_step, frame_size)

    # TODO: didnt fix the cropping issue, now theres not enough cropping...
    #        i dont think this tile ds approach is ther ight way to do it
    # calculate the frame to tile downsampling
    tile_ds = (frame_size - frame_pad) / ntiles

    # generate the locations and sizes of the frames
    frame_locs = loader.get_frame_locs(frame_size, frame_step, tile_size)

    # define an array to hold the result of each frame analysis
    nd = np.ones(
        (frame_locs.shape[1] * ntiles[1], frame_locs.shape[0] * ntiles[0]),
        dtype=np.float,
    )

    # load the frames one by one into memory
    for i in range(frame_locs.shape[0]):
        for j in range(frame_locs.shape[1]):
            frame = loader.load_region(
                frame_locs[i][j], frame_size, pad_color=slide_color)

            # separate the DAB from the Hematoxylin
            _, dab, _ = separator.run(frame, ret=[False, True, False])

            # convert the DAB stain to grayscale
            dab_gray = iproc.rgb_to_gray(dab)

            # calculate neuron density at the specific tile shape
            nd[(j)*ntiles[1]:(j+1)*ntiles[1], (i)*ntiles[0]:(i+1)*ntiles[0]] = \
                iproc.get_neuron_density(loader, dab_gray, tile_size, tile_step)

            # print the progress
            count = i * frame_locs.shape[1] + j
            nframes = frame_locs.shape[0] * frame_locs.shape[1]
            s = 'Progress: %0.2f%%' % (100 * (count+1) / nframes)
            print(s + '\b'*len(s), end='', flush=True)
            count += 1
    print('Done!!')

    # # crop out the padded portions of the frame
    w, h = np.rint(loader.get_dim() / tile_ds).astype(np.int)
    nd = nd[:h, :w]

    # scale the density analysis to a grayscale image
    nd = np.rint(255 * (nd - np.min(nd)) / \
                 (np.max(nd) - np.min(nd))).astype(np.uint8)

    # calculate the histogram of the image
    # NOTE: the log is taken for better thresholding and plotting
    hist, bins = np.histogram(nd, 256)
    hist = np.log(hist, out=np.zeros_like(hist, dtype=np.float), where=hist!=0)

    # get the histogram thresholds
    bckg_tiss = 15
    wm_gm = 75
    lo_hi = 155

    # separate the background from the tissue
    bckg = np.full_like(nd, 0)
    wm = np.full_like(nd, 50)
    nd_bckg_sep = np.where(nd < bckg_tiss, bckg, wm)

    # separate the GM from the WM
    lo_gm = np.full_like(nd, 100)
    nd_wm_sep = np.where(nd < wm_gm, nd_bckg_sep, lo_gm)

    # separate the dense GM from the sparse GM
    hi_gm = np.full_like(nd, 200)
    nd_gm_sep = np.where(nd < lo_hi, nd_wm_sep, hi_gm)

    # detect the tissue from the thumbnail image
    tissue_contours, tb_thresh, tb_tissue_detected = iproc.detect_tissue(
        loader, thumbnail, loader.get_thumbnail_lvl())

    wm_contours, wm_thresh, tb_wm_detected = iproc.detect_wm(
        loader, nd, thumbnail, tissue_contours, tile_ds)

    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(nd, cmap='coolwarm')
    axs[0][0].set_title('Neuron Density - Tile Size: (%d,%d) Microns' % \
                        (tile_size_m[0], tile_size_m[1]))
    axs[1][0].plot(hist, color='black')
    axs[1][0].plot(bckg_tiss, hist[bckg_tiss], '.', color='red', linewidth=5)
    axs[1][0].plot(wm_gm, hist[wm_gm], '.', color='green', linewidth=5)
    axs[1][0].plot(lo_hi, hist[lo_hi], '.', color='blue', linewidth=5)
    axs[1][0].set_title('Log Scaled Histo of Neuron Density')

    axs[0][1].imshow(tb_thresh, cmap='coolwarm')
    axs[0][1].set_title('Thresholded Thumbnail')
    axs[1][1].imshow(tb_tissue_detected, cmap='coolwarm')
    axs[1][1].set_title('Tissue Detected Thumbnail')
    axs[0][1].get_shared_x_axes().join(axs[0][1], axs[1][1])
    axs[0][1].get_shared_y_axes().join(axs[0][1], axs[1][1])

    axs[0][2].imshow(wm_thresh, cmap='coolwarm')
    axs[0][2].set_title('Thresholded Neuron Density')
    axs[1][2].set_title('WM Detected Thumbnail')

    plt.show()


    # TODO: write the detectinos to json
    #       NOTE: script to import to qupath - http://www.andrewjanowczyk.com/exporting-and-re-importing-annotations-from-qupath-for-usage-in-machine-learning/
    #

def plot_cells(img, sep, neuron, non_neuron):

    fig, axs = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(sep)
    ax[1].set_title('Stain Separated')
    ax[2].imshow(neuron)
    ax[2].set_title('Neuronal')
    ax[3].imshow(non_neuron)
    ax[3].set_title('Non-Neuronal')
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()

def plot_stains(img, hem, dab, res):

    fig, axs = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(hem)
    ax[1].set_title('Hematoxylin')
    ax[2].imshow(dab)
    ax[2].set_title('DAB')
    ax[3].imshow(res)
    ax[3].set_title('Residual')
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()

def plot_histos(img, hem, dab, res):

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    ax = axs.ravel()
    ax[0].plot(img[0, :], 'r')
    ax[0].plot(img[1, :], 'g')
    ax[0].plot(img[2, :], 'b')
    ax[0].plot(img[3, :], 'black')
    ax[0].set_title('Original')
    ax[1].plot(hem[0, :], 'r')
    ax[1].plot(hem[1, :], 'g')
    ax[1].plot(hem[2, :], 'b')
    ax[1].plot(hem[3, :], 'black')
    ax[1].set_title('Hematoxylin')
    ax[2].plot(dab[0, :], 'r')
    ax[2].plot(dab[1, :], 'g')
    ax[2].plot(dab[2, :], 'b')
    ax[2].plot(dab[3, :], 'black')
    ax[2].set_title('DAB')
    # ax[3].plot(res[0, :], 'r')
    # ax[3].plot(res[1, :], 'g')
    # ax[3].plot(res[2, :], 'b')
    # ax[3].plot(res[3, :], 'black')
    # ax[3].set_title('Residual')
    for a in ax.ravel():
        a.set_xlim([160, 255])
    fig.tight_layout()


if __name__ == "__main__":
    main(sys.argv)
