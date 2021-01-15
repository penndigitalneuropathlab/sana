
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
    nd = np.rint(255 * nd).astype(np.uint8)

    # detect the tissue from the thumbnail image
    tissue_contours, tb_thresh, tb_tissue_detected = iproc.detect_tissue(
        loader, thumbnail, loader.get_thumbnail_lvl())

    wm_contours, wm_thresh, tb_wm_detected, hist, gm_wm, means, vars = \
        iproc.detect_wm(loader, nd, thumbnail, tissue_contours, tile_ds)

    # plot_color_deconvolution()

    # plot_neuron_density()

    # plot_tissue_detection()

    # plot_neuron_density()

    # plot_wmgm_detection()

    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(nd, cmap='coolwarm')
    axs[0][0].set_title('Neuron Density - Tile Size: (%d,%d) Microns' % \
                        (tile_size_m[0], tile_size_m[1]))
    axs[1][0].plot(hist, color='black')
    axs[1][0].axvline(gm_wm, color='purple', label='WM/GM Density Threshold')
    axs[1][0].arrow(means[0], hist[means[0]], vars[0], 0,
                    color='red', width=10, length_includes_head=True,
                    head_width=300, head_length=1,
                    label='WM Normal Distributino')
    axs[1][0].arrow(means[1], hist[means[1]], vars[1], 0,
                    color='blue', width=10, length_includes_head=True,
                    head_width=300, head_length=1,
                    label='GM Normal Distribution')
    axs[1][0].set_title('Tissue Histogram of Neuron Density')
    axs[1][0].set_xlim([0, 70])

    axs[0][1].imshow(tb_thresh, cmap='coolwarm')
    axs[0][1].set_title('Thresholded Thumbnail')
    axs[1][1].imshow(tb_tissue_detected, cmap='coolwarm')
    axs[1][1].set_title('Tissue Detected Thumbnail')
    axs[0][1].get_shared_x_axes().join(axs[0][1], axs[1][1])
    axs[0][1].get_shared_y_axes().join(axs[0][1], axs[1][1])

    axs[0][2].imshow(wm_thresh, cmap='coolwarm')
    axs[0][2].set_title('Thresholded Neuron Density')
    axs[1][2].imshow(tb_wm_detected, cmap='coolwarm')
    axs[1][2].set_title('GM Detected Thumbnail')

    plt.show()

if __name__ == "__main__":
    main(sys.argv)
