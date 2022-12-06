
# system modules
import os

# installed modules
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import MulticlassClassifier, TangleClassifier

# debugging modules
from matplotlib import pyplot as plt

class AT8Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(AT8Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)

        # either use the cmdl input value or a pre-defined value from before
        # NOTE: this pre-defined value was picked from analyzing multiple slides
        #        in QuPath w/ varying intensities and pathology severity
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90, open_r=0, close_r=0)

        # generate the neuronal and glial severity results
        # self.run_multiclass(odir, params)

        # generate the tangle severity results
        # self.run_tangle(odir, params)

        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')

        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    # TODO: don't want to create the model everytime, should be it's own class maybe?
    def run_tangle(self, odir, params):

        model = TangleClassifier(self.frame)
        probs = model.run()

        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_TANGLE.npy'))
        np.save(ofname, probs)
    #
    # end of run_tangle

    def run_multiclass(self, odir, params):

        model = MulticlassClassifier(self.frame)
        probs = model.run()

        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_MULTICLASS.npy'))
        np.save(ofname, probs)

        # fig, axs = plt.subplots(2,4)
        # axs[0,2].imshow(density[0], cmap='coolwarm')
        # axs[0,3].imshow(density[1], cmap='coolwarm')
        # axs[1,2].imshow(density[2], cmap='coolwarm')
        # axs[1,3].imshow(density[3], cmap='coolwarm')

        # gs = axs[0,0].get_gridspec()
        # for ax in axs[0:2, 0:2].flatten():
        #     ax.remove()
        # axbig = fig.add_subplot(gs[0:2,0:2])
        # axbig.imshow(self.frame.img)
        # plt.show()

        # # get the prob. maps as a pos. class minus the bckg class
        # neuronal = x_cpool[0] - x_cpool[3]
        # glial = x_cpool[1] - x_cpool[3]
        # projections = x_cpool[2] - x_cpool[3]

        # # threshold the data

        # # run the %AO algo on the thresholded images
        # neuronal_results = self.run_ao(neuronal)
        # glial_results = self.run_ao(glial)
        # projections_results = self.run_ao(projections)

        # # store the results
        # # parmas.data[''] = results['']
    #
    # end of run_wildcat

#
# end of AT8Processor

#
# end of file
