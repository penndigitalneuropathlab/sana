
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, overlay_thresh
from sana_tractography import STA
from sana_thresholds import max_dev, kittler
from processors.HDAB_processor import HDABProcessor

# this is a H-DAB stain which stains for -------
# it is a specific antibody so the threshold is lenient
# we also perform structure tensor analysis to identify
# individual ------ in various directions in the tissue
class MBPProcessor(HDABProcessor):
    def __init__(self, fname, frame):
        super(MBPProcessor, self).__init__(fname, frame)
    #
    # end of constructor

    def run(self, odir, params, main_roi, sub_rois=[]):

        self.mask_frame(main_roi, sub_rois)
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params)

        # generate the vertical fibers AO
        self.run_vertical_ao(odir, params)

        # generate the horizontal fibers AO
        self.run_horizontal_ao(odir, params)
        
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

    def run_vertical_ao(self, odir, params):

        # for vertical
        sigma = (10,100)
        self.vert_sta = STA(sigma)
        self.vert_sta.run(self.dab_norm)

        # get the distance from 90, then inverse it
        # NOTE: this maps 0 and 180 -> 0, 90 -> 90
        self.vert_sta.ang = (90 - np.abs(90 - self.vert_sta.ang))
        self.vert_sta.ang /= 90

        # create the probability map
        # NOTE: we are essentially scaling the DAB prob. by the coh and ang
        self.vert_prob = Frame(self.vert_sta.ang,
                               self.frame.lvl, self.frame.converter)
        self.vert_prob.img *= self.dab_norm.img
        self.vert_prob.img *= self.vert_sta.coh
        
        # rescale the prob. map to 8 bit pixels
        # TODO: this could be problematic, i dont like the scale by max
        self.vert_prob.img /= np.max(self.vert_prob.img)
        self.vert_prob.img = (255 * self.vert_prob.img).astype(np.uint8)

        # get the histogram and threshold
        self.vert_hist = self.vert_prob.histogram()
        self.vert_threshold = max_dev(self.vert_hist)

        # threshold the prob. map
        self.vert_thresh = self.vert_prob.copy()
        self.vert_thresh.threshold(self.vert_threshold, 0, 255)

        # run the AO process
        results = self.run_ao(self.vert_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']        
        params.data['vert_fibers_ao'] = results['ao']
        params.data['vert_fibers_sub_aos'] = results['sub_aos']
        params.data['vert_threshold'] = self.vert_thresh

        # create the output directory
        odir = sana_io.create_odir(odir, 'vert_ao')

        # save the images used in processing
        self.vert_overlay = overlay_thresh(
            self.frame, self.vert_thresh)
        self.save_frame(odir, Frame(self.vert_sta.coh), 'COH')
        self.save_frame(odir, Frame(self.vert_sta.ang), 'ANG')        
        self.save_frame(odir, self.vert_prob, 'PROB')
        self.save_frame(odir, self.vert_thresh, 'THRESH')
        self.save_frame(odir, self.vert_overlay, 'QC')
    #
    # end of run_vertical_ao

    def run_horizontal_ao(self, odir, params):

        sigma = (100,10)
        self.horz_sta = STA(sigma)
        self.horz_sta.run(self.dab_norm)
        
        # get the distance from 90
        # NOTE: this maps 0 and 180 -> 90, 90 -> 0
        self.horz_sta.ang = (90 - self.horz_sta.ang)
        self.horz_sta.ang /= 90
        
        # create the probability map
        # NOTE: we are essentially scaling the DAB prob. by the coh and ang
        # TODO: this part is redundant with vert_ao
        self.horz_prob = Frame(self.horz_sta.ang,
                               self.frame.lvl, self.frame.converter)
        self.horz_prob.img *= self.dab_norm.img
        self.horz_prob.img *= self.horz_sta.coh
        
        # rescale the prob. map to 8 bit pixels
        # TODO: this could be problematic, i dont like the scale by max
        self.horz_prob.img /= np.max(self.horz_prob.img)
        self.horz_prob.img = (255 * self.horz_prob.img).astype(np.uint8)

        # get the histogram and threshold
        self.horz_hist = self.horz_prob.histogram()
        self.horz_threshold = max_dev(self.horz_hist)

        # threshold the prob. map
        self.horz_thresh = self.horz_prob.copy()
        self.horz_thresh.threshold(self.horz_threshold, 0, 255)

        # run the AO process
        results = self.run_ao(self.horz_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']        
        params.data['horz_fibers_ao'] = results['ao']
        params.data['horz_fibers_sub_aos'] = results['sub_aos']
        params.data['horz_threshold'] = self.horz_thresh

        # create the output directory
        odir = sana_io.create_odir(odir, 'horz_ao')

        # save the images used in processing
        self.horz_overlay = overlay_thresh(
            self.frame, self.horz_thresh)
        self.save_frame(odir, Frame(self.horz_sta.coh), 'COH')
        self.save_frame(odir, Frame(self.horz_sta.ang), 'ANG')        
        self.save_frame(odir, self.horz_prob, 'PROB')
        self.save_frame(odir, self.horz_thresh, 'THRESH')
        self.save_frame(odir, self.horz_overlay, 'QC')
    #
    # end of run_horizontal_ao
#
# end of MBPProcessor

#
# end of file
