
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, overlay_thresh
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler

# debugging modules
from matplotlib import pyplot as plt

# instantiates a Processor object based on the antibody of the svs slide
def get_processor(fname, frame):
    antibody = sana_io.get_antibody(fname)
    if antibody == 'NeuN':
        return NeuNProcessor(fname, frame)
    if antibody == 'parvalbumin':
        return parvalbuminProcessor(fname, frame)
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
    
    # generic function to calculate %AO of a thresholded frame
    # 1) gets the %AO of the main_roi
    # 2) gets the %AO of the sub_rois provided (if any)
    #
    # NOTE: the frame and rois must be in the same coord. system
    # TODO: any way to check the above??, could check if theres no overlap between frame nad mask
    #       IMPORTANT! not actually AO=0, if the frame is black you would still get AO=0
    def run_ao(self, orig_frame, main_roi, sub_rois=[]):

        # create a copy to not disturb the original data
        # TODO: need to check that it is thresholded! set a flag in frame.threshold()
        frame = orig_frame.copy()
        
        # generate the main mask
        main_mask = create_mask(
            [main_roi],
            frame.size(), frame.lvl, frame.converter,
            x=0, y=255, holes=[]
        )
        
        # generate the sub masks
        sub_masks = []
        for i in range(len(sub_rois)):
            mask = create_mask(
                [sub_rois[i]],
                frame.size(), frame.lvl, frame.converter,
                x=0, y=255, holes=[]
            )
            sub_masks.append(mask)

        # apply the main mask to the image
        frame.mask(main_mask)

        # get the total area of the roi
        area = np.sum(main_mask.img / 255)

        # get the pos. area in the frame
        pos = np.sum(frame.img / 255)

        # calculate %AO of the main roi
        ao = pos / area
        
        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = [], []
        for sub_mask in sub_masks:
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
            'main_mask': main_mask, 'sub_masks': sub_masks,
        }
        return ret
    #
    # end of run_ao
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
#
# end of HDABProcessor

# this is a H-DAB stain which stains for -------
# it not a very specific antibody, therefore the thresholds are
# more strict than usual to account for that
class parvalbuminProcessor(HDABProcessor):
    def __init__(self, fname, frame):
        super(parvalbuminProcessor, self).__init__(fname, frame)
    #
    # end of constructor

    def run(self, odir, params, main_roi, sub_rois=[]):

        # generate the manually curated AO results
        manual_results = self.run_manual_ao(main_roi, sub_rois)

        # generate the auto AO results
        auto_results = self.run_auto_ao(main_roi, sub_rois)

        # store the results of the AO processes
        params.data['manual_stain_threshold'] = self.manual_dab_threshold
        params.data['auto_stain_threshold'] = self.auto_dab_norm_threshold
        params.data['manual_ao'] = manual_results['ao']
        params.data['auto_ao'] = auto_results['ao']
        params.data['area'] = auto_results['area']
        params.data['manual_sub_aos'] = manual_results['sub_aos']
        params.data['auto_sub_aos'] = auto_results['sub_aos']
        params.data['sub_areas'] = auto_results['sub_areas']

        # save the params IO to a file
        params_f = sana_io.create_filepath(self.fname, ext='.csv', fpath=odir)
        params.write_data(params_f)

        # save the original frame
        orig_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='ORIG', fpath=odir)
        self.frame.save(orig_f)

        # save the DAB and HEM images
        dab_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='DAB', fpath=odir)
        hem_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='HEM', fpath=odir)
        self.dab.save(dab_f)
        self.hem.save(hem_f)

        # create the output sub directories
        manual_odir = sana_io.create_odir(odir, 'manual_ao')
        auto_odir = sana_io.create_odir(odir, 'auto_ao')
        
        # save the probability map
        dab_prob_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='PROB', fpath=auto_odir)
        self.dab_norm.save(dab_prob_f)

        # save the thresholded images
        manual_dab_thresh_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='THRESH', fpath=manual_odir)
        auto_dab_thresh_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='THRESH', fpath=auto_odir)
        self.manual_dab_thresh.save(manual_dab_thresh_f)
        self.auto_dab_norm_thresh.save(auto_dab_thresh_f)

        # create and save the overlays
        self.manual_overlay = overlay_thresh(self.frame, self.manual_dab_thresh)
        self.auto_overlay = overlay_thresh(self.frame, self.auto_dab_norm_thresh)
        manual_overlay_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='QC', fpath=manual_odir)
        auto_overlay_f = sana_io.create_filepath(
            self.fname, ext='.png', suffix='QC', fpath=auto_odir)
        self.manual_overlay.save(manual_overlay_f)
        self.auto_overlay.save(auto_overlay_f)
        
        # TODO: save the masks?
    #
    # end of run
    
    # performs a simple threshold using a manually selected cut off point
    # then runs the %AO process
    def run_manual_ao(self, main_roi, sub_rois=[]):
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 1.0 in QuPath, this
        #       value is calculated by that
        self.manual_dab_threshold = 127

        # apply the thresholding
        self.manual_dab_thresh = self.dab.copy()
        self.manual_dab_thresh.threshold(self.manual_dab_threshold, 0, 255)

        return self.run_ao(self.manual_dab_thresh, main_roi, sub_rois)
    #
    # end of run_manual_ao

    # performs normalization, smoothing, and histogram
    # TODO: this is pretty generic, probably just need a auto_ao in HDABProcessor
    #       with a scale parameter, and turn on/off smoothing?
    def run_auto_ao(self, main_roi, sub_rois=[]):
        
        # normalize the image
        self.dab_norm = mean_normalize(self.dab)

        # smooth the image
        self.dab_norm.anisodiff()
        
        # get the histograms
        self.dab_hist = self.dab.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        # get the stain threshold
        # NOTE: we want strict thresholding here because PV is not very specific
        self.auto_dab_threshold = max_dev(self.dab_hist, scale=0.15)                
        self.auto_dab_norm_threshold = max_dev(self.dab_norm_hist, scale=0.15)        

        # apply the thresholding
        self.auto_dab_norm_thresh = self.dab_norm.copy()
        self.auto_dab_norm_thresh.threshold(self.auto_dab_norm_threshold, 0, 255)

        return self.run_ao(self.auto_dab_norm_thresh, main_roi, sub_rois)
    #
    # end of run_auto_ao

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
