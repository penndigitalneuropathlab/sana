
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import imageio as iio
import imageproc as iproc

SRC = "~/neuropath/data/layer_segmentation/images/"
DST = "~/neuropath/src/qupath-automatic-gm-segmentation/example_images/tissue_detection"

def main(argv):

    # retrieve the directory locations
    if '-i' not in argv:
        src = SRC
    else:
        src = argv[argv.index('-i')+1]
    if '-o' not in argv:
        dst = DST
    else:
        dst = argv[argv.index('-o')+1]
    src = iio.get_fullpath(src)
    dst = iio.get_fullpath(dst)

    ifiles = [os.path.join(src, f) for f in os.listdir(src)]
    ofiles = [f.replace(src, dst).replace('.svs', '.png') for f in ifiles]

    lvl = 2
    tile_size = np.array((400,400))
    tile_step = np.array((100,100))
    for ifile, ofile in zip(ifiles, ofiles):

        # TODO: write the tissue contours to a json file
        loader = iio.SVSLoader(ifile)
        tissue_contours, _, img, _, _ = iproc.run_tissue_detection(loader)

        # plotting...
        fig, axs = plt.subplots()
        axs.imshow(img)
        axs.grid(False)
        axs.axis('off')
        plt.savefig(ofile, dpi=1000)

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
