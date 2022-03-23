
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler

# debugging modules
from matplotlib import pyplot as plt

def get_processor(fname, frame):
    antibody = sana_io.get_antibody(fname)
    if antibody == 'NeuN':
        return NeuNProcessor(frame)
    if antibody == 'parvalbumin':
        return parvalbuminProcessor(frame)

class Processor:
    def __init__(self, frame):
        self.frame = frame

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
                sub_rois[i],
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
        ao = pos / total_area
        
        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = []
        for sub_mask in sub_masks:
            tmp_frame = frame.copy()
            tmp_frame.mask(sub_mask)
            sub_area = np.sum(sub_mask.img / 255)
            sub_pos = np.sum(tmp_frame.img / 255)
            sub_ao = sub_pos / sub_area
            sub_aos.append(sub_aos)
            sub_areas.append(sub_area)

        # finally, return the results
        ret = {
            'ao': ao, 'area': area,
            'sub_aos': sub_aos, 'sub_areas': sub_areas,
        }
        return ret
    #
    # end of run_ao
#
# end of Processor

class NeuNProcessor(Processor):
    def __init__(self, frame):
        super(NeuNProcessor, self).__init__(frame)

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

class parvalbuminProcessor(Processor):
    def __init__(self, frame):
        super(parvalbuminProcessor, self).__init__(frame)
        self.frame = frame
        
        # prepare the stain separator
        self.ss = StainSeparator('H-DAB')

        # separate out the HEM and DAB stains
        self.stains = self.ss.run(frame.img)
        self.hem = Frame(self.stains[:,:,0], frame.lvl, frame.converter)
        self.dab = Frame(self.stains[:,:,1], frame.lvl, frame.converter)

        # rescale the OD stains to 8 bit pixel values
        # NOTE: this uses the physical min/max of the stains based
        #       on the stain vector used
        self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])
    #
    # end of constructor
    
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

    def run_auto_ao(self, main_roi, sub_rois=[]):
        
        # normalize the images
        self.hem_norm = mean_normalize(self.hem)
        self.dab_norm = mean_normalize(self.dab)

        # smooth the images
        self.hem_norm.anisodiff()
        self.dab_norm.anisodiff()
        
        # get the histograms of both stains
        self.hem_hist = self.hem.histogram()
        self.dab_hist = self.dab.histogram()
        self.hem_norm_hist = self.hem_norm.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        print(max_dev(self.dab_hist, scale=1.0))
        print(max_dev(self.dab_hist, scale=0.15))

        # get the stain thresholds
        # NOTE: we want strict thresholding here because PV is not very specific
        # TODO: also do a manual thresh of DAB=127, no norm!!!!
        self.hem_threshold = kittler(self.hem_hist)
        self.dab_threshold = max_dev(self.dab_hist, scale=0.15)
        self.hem_norm_threshold = max_dev(self.hem_norm_hist) # TODO: anyway to make this less strict?
        self.dab_norm_threshold = max_dev(self.dab_norm_hist, scale=0.15)

        # TODO: finally, run the run_ao() function
    #
    # end of run_auto_ao
#
# end of parvalbuminProcessor
