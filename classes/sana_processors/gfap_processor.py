 
# system modules

# installed modules

# custom modules
from sana_processors.HDAB_processor import HDABProcessor

# debugging modules
from matplotlib import pyplot as plt

class GFAPProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(GFAPProcessor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=0.5, mx=255)

        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run
#
# end of GFAPProcessor

#
# end of file
