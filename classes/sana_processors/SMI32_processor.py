
# custom modules
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor

class SMI32Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(SMI32Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    # TODO: might not even need run?
    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.4 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 99
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        # NOTE: this value is a little strict to remove some background
        # TODO: do we want to run some opening filter?
        self.run_auto_ao(odir, params, scale=0.7)

        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run    
#
# end of SMI32Processor

#
# end of file
