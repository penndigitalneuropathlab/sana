
# system modules

# installed modules

# custom modules
from sana_processors import HDABProcessor

# debugging modules

class LFBProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger):
        super(LFBProcessor, self).__init__(fname, frame, logger)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):
        self.generate_masks(main_roi, sub_rois)

        self.run_cell_density(odir, params)
    #
    # end of run

    def run_cell_density(self, odir, params):   
        pass
    #
    # end of run_cell_density
#
# end of R13Processor

#
# end of file
