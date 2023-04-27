
# custom modules
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor

class calretininProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(calretininProcessor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=0.3)

        # save the original frame
        # TODO: where should this go? shouldn't be in every run()...
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run    

    # TODO: this should prolly be in HDABProcessor?
    #         there might be some stains that need specific processing, but
    #         most will be generic
    def run_hem(self):
        self.hem_norm = mean_normalize(self.hem)        
        self.hem_norm.anisodiff()        
        self.hem_hist = self.hem.histogram()
        self.hem_norm_hist = self.hem_norm.histogram()        
        self.hem_threshold = kittler(self.hem_hist)
        # TODO: anywayto make this less strict?
        self.hem_norm_threshold = max_dev(self.hem_norm_hist) 
    #
    # end of run_hem
#
# end of parvalbuminProcessor

#
# end of file
