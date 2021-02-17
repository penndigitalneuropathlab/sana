
import os
import sys
import argparse
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sana_loader import Loader
from sana_thresholder import TissueThresholder
from sana_detector import TissueDetector

from sklearn.mixture import GaussianMixture as GMM

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

    # threshold the frame and generate the tissue mask
    thresholder = TissueThresholder(frame, blur=5, mi=180, mx=255)

    models = []
    N = list(range(2, 100))
    for k in N:
        models.append(GMM(k, covariance_type='full').fit(thresholder.data))

    plt.plot(N, [m.bic(thresholder.data) for m in models], label='BIC')
    plt.plot(N, [m.aic(thresholder.data) for m in models], label='AIC')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
