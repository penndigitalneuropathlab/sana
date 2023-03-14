
# system modules
import os

# installed modules
import numpy as np

# custom modules
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor

# debugging modules
from matplotlib import pyplot as plt

class TDP43MPProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(TDP43MPProcessor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90)
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run
#
# end of TDP43MPProcessor

#
# end of file
