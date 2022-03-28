
# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, overlay_thresh
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_antibody_processor import Processor

# generic Processor for H-DAB stained slides
# performs stain separation and rescales the data to 8 bit pixels
class HDABProcessor(Processor):
    def __init__(self, fname, frame):
        super(HDABProcessor, self).__init__(fname, frame)
        
        # prepare the stain separator
        self.ss = StainSeparator('H-DAB')

        # separate out the HEM and DAB stains
        self.stains = self.ss.run(self.frame.img)
        self.hem = Frame(self.stains[:,:,0], frame.lvl, frame.converter)
        self.dab = Frame(self.stains[:,:,1], frame.lvl, frame.converter)

        # rescale the OD stains to 8 bit pixel values
        # NOTE: this uses the physical min/max of the stains based
        #       on the stain vector used
        self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])
    #
    # end of constructor

    # performs a simple threshold using a manually selected cut off point
    # then runs the %AO process
    def run_manual_ao(self, odir, params):

        # apply the thresholding
        self.manual_dab_thresh = self.dab.copy()
        self.manual_dab_thresh.threshold(self.manual_dab_threshold, 0, 255)

        results = self.run_ao(self.manual_dab_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']
        params.data['manual_ao'] = results['ao']
        params.data['manual_sub_aos'] = results['sub_aos']
        params.data['manual_stain_threshold'] = self.manual_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir, 'manual_ao')

        # save the images used in processing
        self.manual_overlay = overlay_thresh(
            self.frame, self.manual_dab_thresh)        
        self.save_frame(odir, self.manual_dab_thresh, 'THRESH')
        self.save_frame(odir, self.manual_overlay, 'QC')        
    #
    # end of run_manual_ao

    # performs normalization, smoothing, and histogram
    # TODO: rename scale to something better
    # TODO: add switches to turn off/on mean_norm, anisodiff, morph
    def run_auto_ao(self, odir, params, scale=1.0):
        
        # normalize the image
        # TODO: this is failing because we're masking too early
        self.dab_norm = mean_normalize(self.dab)

        # smooth the image
        self.dab_norm.anisodiff()
        
        # get the histograms
        self.dab_hist = self.dab.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        # get the stain threshold
        # NOTE: we want strict thresholding here because PV is not very specific
        self.auto_dab_threshold = max_dev(self.dab_hist, scale=scale)
        self.auto_dab_norm_threshold = max_dev(self.dab_norm_hist, scale=scale)

        # apply the thresholding
        self.auto_dab_norm_thresh = self.dab_norm.copy()
        self.auto_dab_norm_thresh.threshold(self.auto_dab_norm_threshold, 0, 255)

        # run the AO process
        results = self.run_ao(self.auto_dab_norm_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']        
        params.data['auto_ao'] = results['ao']
        params.data['auto_sub_aos'] = results['sub_aos']
        params.data['auto_stain_threshold'] = self.auto_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir, 'auto_ao')

        # save the images used in processing
        self.auto_overlay = overlay_thresh(
            self.frame, self.auto_dab_norm_thresh)        
        self.save_frame(odir, self.dab_norm, 'PROB')
        self.save_frame(odir, self.auto_dab_norm_thresh, 'THRESH')
        self.save_frame(odir, self.auto_overlay, 'QC')
    #
    # end of run_auto_ao    
#
# end of HDABProcessor

#
# end of file
