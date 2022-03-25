
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, overlay_thresh
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_tractography import STA

# debugging modules
from matplotlib import pyplot as plt

# instantiates a Processor object based on the antibody of the svs slide
def get_processor(fname, frame):
    antibody = sana_io.get_antibody(fname)
    if antibody == 'NeuN':
        return NeuNProcessor(fname, frame)
    if antibody == 'parvalbumin':
        return parvalbuminProcessor(fname, frame)
    if antibody == 'SMI94':
        return MBPProcessor(fname, frame)
#
# end of get_processor

# generic processor class, sets the main attributes and holds
# functions for generating data from processed Frames
class Processor:
    def __init__(self, fname, frame):
        self.fname = fname
        self.frame = frame
    #
    # end of constructor

    def mask_frame(self, main_roi, sub_rois=[]):
        # generate the main mask
        self.main_mask = create_mask(
            [main_roi],
            self.frame.size(), self.frame.lvl, self.frame.converter,
            x=0, y=255, holes=[]
        )
        
        # generate the sub masks
        self.sub_masks = []
        for i in range(len(sub_rois)):
            mask = create_mask(
                [sub_rois[i]],
                self.frame.size(), self.frame.lvl, self.frame.converter,
                x=0, y=255, holes=[]
            )
            self.sub_masks.append(mask)
        #
        # end of sub_masks loop

        # finally, mask the frame by the main ROI
        self.frame.mask(self.main_mask)
        self.dab.mask(self.main_mask)
        self.hem.mask(self.main_mask)        
    #
    # end of gen_masks
    
    # generic function to calculate %AO of a thresholded frame
    # 1) gets the %AO of the main_roi
    # 2) gets the %AO of the sub_rois provided (if any)
    #
    # NOTE: the frame and rois must be in the same coord. system
    # TODO: any way to check the above??, could check if theres no overlap between frame nad mask
    #       IMPORTANT! not actually AO=0, if the frame is black you would still get AO=0
    def run_ao(self, orig_frame):

        # create a copy to not disturb the original data
        # TODO: need to check that it is thresholded! set a flag in frame.threshold()
        frame = orig_frame.copy()
        
        # get the total area of the roi
        area = np.sum(self.main_mask.img / 255)

        # get the pos. area in the frame
        pos = np.sum(frame.img / 255)

        # calculate %AO of the main roi
        ao = pos / area
        
        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = [], []
        for sub_mask in self.sub_masks:
            tmp_frame = frame.copy()
            tmp_frame.mask(sub_mask)
            sub_area = np.sum(sub_mask.img / 255)
            sub_pos = np.sum(tmp_frame.img / 255)
            sub_ao = sub_pos / sub_area
            sub_aos.append(sub_ao)
            sub_areas.append(sub_area)

        # finally, return the results
        ret = {
            'ao': ao, 'area': area,
            'sub_aos': sub_aos, 'sub_areas': sub_areas,
        }
        return ret
    #
    # end of run_ao

    def save_frame(self, odir, frame, suffix):
        fpath = sana_io.create_filepath(
            self.fname, ext='.png', suffix=suffix, fpath=odir)
        frame.save(fpath)
    #
    # end of save_frame

    def save_params(self, odir, params):
        fpath = sana_io.create_filepath(self.fname, ext='.csv', fpath=odir)
        params.write_data(fpath)
    #
    # end of save_params
#
# end of Processor

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
        # TODO: check the math!
        # TODO: this doesn't work
        self.manual_dab_threshold = 35

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
        self.vert_sta.run(self.dab_norm, debug=True)

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
        params.data['vert_fibers_aos'] = results['sub_aos']
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
        self.horz_sta.run(self.dab_norm, debug=True)
        
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
        params.data['horz_fibers_aos'] = results['sub_aos']
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

# this is a H-DAB stain which stains for -------
# it not a very specific antibody, therefore the thresholds are
# more strict than usual to account for that
class parvalbuminProcessor(HDABProcessor):
    def __init__(self, fname, frame):
        super(parvalbuminProcessor, self).__init__(fname, frame)
    #
    # end of constructor

    # TODO: might not even need run?
    def run(self, odir, params, main_roi, sub_rois=[]):

        self.mask_frame(main_roi, sub_rois)
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 1.0 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 127
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params)

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

class NeuNProcessor(Processor):
    def __init__(self, fname, frame):
        super(NeuNProcessor, self).__init__(fname, frame)

        # separate the stains from the RGB frame
        self.ss = StainSeparator('H-DAB')
        self.stains = self.ss.run(frame.img)
        self.hem = Frame(self.stains[:,:,0], frame.lvl, frame.converter)
        self.dab = Frame(self.stains[:,:,1], frame.lvl, frame.converter)
        self.hemdab = np.copy(self.stains)
        self.hemdab[:,:,2] = 0
        self.hemdab = Frame(self.ss.combine_stains(self.hemdab), frame.lvl, frame.converter)

        # rescale the OD to 8 bit pixel values
        self.hem.rescale()
        self.dab.rescale()
        self.hemdab.to_gray()
        self.hemdab.img = 255 - self.hemdab.img

        # TODO: keep plotting after everyone processing step to make sure the hists look good!!!!
        fig, axs = plt.subplots(2,3)
        axs = axs.ravel()
        axs[0].imshow(self.hem.img)
        axs[1].imshow(self.dab.img)
        axs[2].imshow(self.hemdab.img)
        axs[3].plot(self.hem.histogram())
        axs[4].plot(self.dab.histogram())
        axs[5].plot(self.hemdab.histogram())        
        plt.show()
        exit()
#
# end of NeuNProcessor
