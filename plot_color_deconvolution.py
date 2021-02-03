
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
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
    plot(NAME % "Color_Denconvolution_Thumbnail", f, lvl)

    lvl = 1
    loc = np.array((14000,9500))
    size = np.array((1000, 800))
    plot(NAME % "Color_Denconvolution_Zoomed", f, lvl, loc, size)

def plot(name, f, lvl, loc=None, size=None):

    separator = iproc.StainSeparator('H-DAB')

    loader = iio.SVSLoader(f)

    loader.set_lvl(lvl)

    if loc is None:
        img = loader.load_thumbnail()
    else:
        loc = loader.micron_to_px(loc)
        size = loader.micron_to_px(size)
        img = loader.load_region(loc, size)

    hem, dab, res, xdh = separator.run(img, rescale=True)

    # plotting...
    fig, axs = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(hem)
    ax[1].set_title('Hematoxylin')
    ax[2].imshow(dab)
    ax[2].set_title('DAB')
    ax[3].imshow(xdh)
    ax[3].set_title('Stain Separated Image (Rescaled)')
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()
    plt.savefig(iio.get_fullpath(name), dpi=500)

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
