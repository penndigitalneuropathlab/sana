
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, overlay_thresh
from sana_tractography import STA
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor

# debugging modules
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# this is a H-DAB stain which stains for -------
# it is a specific antibody so the threshold is lenient
# we also perform structure tensor analysis to identify
# individual ------ in various directions in the tissue
class MBPProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(MBPProcessor, self).__init__(fname, frame, logger, **kwargs)
        self.debug = debug
        self.debug_fibers = debug_fibers
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
        self.run_auto_ao(odir, params, scale=1.0)

        self.vert_sigma = (5,20)        
        self.horz_sigma = (20,5)
        
        # generate the vertical fibers AO
        self.run_vertical_ao(odir, params)

        # generate the horizontal fibers AO
        self.run_horizontal_ao(odir, params)

        if self.debug_fibers:
            self.show_fibers()

        self.fibers_overlay = self.frame.copy()
        self.fibers_overlay = overlay_thresh(
            self.fibers_overlay, self.horz_thresh, alpha=0.4, color='blue')
        self.fibers_overlay = overlay_thresh(
            self.fibers_overlay, self.vert_thresh, alpha=0.4, color='red')
            
        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')

        self.save_frame(odir, self.fibers_overlay, 'FIBERS')
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    def run_vertical_ao(self, odir, params):
        
        # for vertical
        self.vert_sta = STA(self.vert_sigma)
        self.vert_sta.run(self.dab)

        # get the distance from 90, then inverse it
        # NOTE: this maps 0 and 180 -> 0, 90 -> 90
        self.vert_sta.ang = (90 - np.abs(90 - self.vert_sta.ang))
        self.vert_sta.ang /= 90
        
        # create the probability map
        # NOTE: we are essentially scaling the DAB prob. by the coh and ang
        self.vert_prob = Frame(self.vert_sta.ang.copy(),
                               self.frame.lvl, self.frame.converter)
        self.vert_prob.img *= self.dab_norm.img
        self.vert_prob.img *= self.vert_sta.coh
        
        # rescale the prob. map to 8 bit pixels
        # TODO: this could be problematic, i dont like the scale by max
        self.vert_prob.img /= np.max(self.vert_prob.img)
        self.vert_prob.img = (255 * self.vert_prob.img).astype(np.uint8)

        # get the histogram and threshold
        self.vert_hist = self.vert_prob.histogram()
        self.vert_threshold = max_dev(self.vert_hist, mx=80, debug=self.debug_fibers)

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

        self.horz_sta = STA(self.horz_sigma)
        self.horz_sta.run(self.dab)
        
        # get the distance from 90
        # NOTE: this maps 0 and 180 -> 90, 90 -> 0
        self.horz_sta.ang = np.abs(90 - self.horz_sta.ang)
        self.horz_sta.ang /= 90
        
        # create the probability map
        # NOTE: we are essentially scaling the DAB prob. by the coh and ang
        # TODO: this part is redundant with vert_ao
        self.horz_prob = Frame(self.horz_sta.ang.copy(),
                               self.frame.lvl, self.frame.converter)
        self.horz_prob.img *= self.dab_norm.img
        self.horz_prob.img *= self.horz_sta.coh
        
        # rescale the prob. map to 8 bit pixels
        # TODO: this could be problematic, i dont like the scale by max
        self.horz_prob.img /= np.max(self.horz_prob.img)
        self.horz_prob.img = (255 * self.horz_prob.img).astype(np.uint8)

        # get the histogram and threshold
        self.horz_hist = self.horz_prob.histogram()
        self.horz_threshold = max_dev(self.horz_hist, mx=80, debug=self.debug_fibers)

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

    def show_fibers(self):
        fig, axs = plt.subplots(1,5, sharex=True, sharey=True)
        axs[0].imshow(self.frame.img)
        axs[1].imshow(self.vert_overlay.img)
        axs[2].imshow(self.vert_sta.coh)
        axs[2].set_title('coherence')
        axs[3].imshow(self.vert_sta.ang)
        axs[3].set_title('angle, closeness to 90 degrees')
        vmax = np.mean(self.vert_prob.img) + 4*np.std(self.vert_prob.img)
        print(vmax, flush=True)
        axs[4].imshow(self.vert_prob.img, vmin=0, vmax=vmax)
        axs[4].set_title('prob map')        
        fig.suptitle('Vertical')
            
        fig, axs = plt.subplots(1,5, sharex=True, sharey=True)
        axs[0].imshow(self.frame.img)
        axs[1].imshow(self.horz_overlay.img)
        axs[2].imshow(self.horz_sta.coh)
        axs[2].set_title('coherence')
        axs[3].imshow(self.horz_sta.ang)
        axs[3].set_title('angle, distance from 90 degrees')
        vmax = np.mean(self.horz_prob.img) + 4*np.std(self.horz_prob.img)
        print(vmax)        
        axs[4].imshow(self.horz_prob.img, vmax=vmax)
        axs[4].set_title('prob map')
        fig.suptitle('Horizontal')
            
        plt.show()
    #
    # end of show_fibers
#
# end of MBPProcessor

#
# end of file
