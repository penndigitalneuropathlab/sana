
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

    lvl = 2
    tsizes = np.array([
        (400,400),
        (100,100),
    ])
    tsteps = np.array([
        (50,50),
        (50,50),
    ])

    fsize = np.array((2000,2000))
    fstep = np.copy(fsize)
    nds = []
    for tsize, tstep in zip(tsizes, tsteps):
        loader = iio.SVSLoader(f)
        loader.set_lvl(lvl)
        loader.set_frame_dims(fsize, fstep)
        loader.set_tile_dims(tsize, tstep)
        nds.append(iproc.run_neuron_density(loader))

    fname = NAME % "Neuron_Density_Tile_Sizes"

    # perform a kron upsampling so the images are the same dimensions
    ds = nds[0].shape[0] / nds[1].shape[0]
    n = int(round(1/ds))
    nds[0] = np.kron(nds[0], np.ones((n, n)))

    # plotting...
    fig, axs = plt.subplots(1, len(nds), figsize=(7, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    for i in range(len(nds)):
        ax[i].imshow(nds[i], cmap='coolwarm')
        ax[i].set_title('Neuron Density - Tile Size (%d, %d)' % \
                        tuple(tsizes[i]))
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()
    plt.savefig(iio.get_fullpath(fname), dpi=1000)

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
