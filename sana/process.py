
# installed modules
import numpy as np
from matplotlib import pyplot as plt
import cv2
import skfmm
from tqdm import tqdm
import skimage.feature

# sana modules
import sana.color_deconvolution
import sana.image
import sana.threshold
import sana.geo
import sana.filter

class Processor:
    """
    Generic antibody processor class. This class sets ROI and mask attributes
    :param logger: Logger object
    :param frame: input RGB image to process
    :param main_roi: ROI which defines the pixels within the frame to process
    :param sub_rois: defines sub-regions to process within the main_roi
    :param exclusion_rois: defines pixels which should not be processed within the main_roi
    """
    def __init__(
            self,
            logger: sana.logging.Logger,
            frame: sana.image.Frame,
            main_roi: sana.geo.Polygon,
            sub_rois: [sana.geo.Polygon]=[],
            exclusion_rois: [sana.geo.Polygon]=[],
    ):
        self.logger = logger
        self.frame = frame

        # generate the main mask
        self.main_roi = main_roi
        self.main_mask = sana.image.create_mask_like(self.frame, [self.main_roi])

        # generate the sub masks
        self.sub_rois = []
        self.sub_masks = []
        for sub_roi in sub_rois:
            if not sub_roi is None:
                self.sub_rois.append(sub_roi)
                self.sub_masks.append(sana.image.create_mask_like(self.frame, [sub_roi]))
            else:
                self.sub_rois.append(None)
                self.sub_masks.append(None)

        # generate the exclusion mask
        self.exclusion_rois = exclusion_rois
        self.exclusion_mask = sana.image.create_mask_like(self.frame, self.exclusion_rois)
        self.valid_mask = self.exclusion_mask.copy()
        self.valid_mask.img = 1 - self.valid_mask.img

        # apply the exclusion mask to the other masks
        self.main_mask.mask(self.valid_mask)
        for sub_mask in self.sub_masks:
            if not sub_mask is None:
                sub_mask.mask(self.valid_mask)

    def classify_pixels(self, frame, threshold, mask=None, morphology_filters=[], debug=True):
        """
        Classifies pixels as foreground/background based on a threshold and morphology filters
        :param frame: grayscale image to classify
        :param threshold: threshold for input frame
        :param closing_radius: radius of morphological closing kernel
        :param opening_radius: radius of morphological opening kernel
        """
        if self.logger.debug_level == 'full':
            fig, axs = plt.subplots(1, 2+len(morphology_filters), sharex=True, sharey=True, figsize=(20,10))
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            axs[0].set_title('Original')

        # apply the threshold
        frame.threshold(threshold)

        # DEBUG:
        if self.logger.debug_level == 'full':
            axs[1].imshow(frame.img)
            axs[1].set_title('Threshold')

        # apply each filter
        for i, morphology_filter in enumerate(morphology_filters):
            frame.apply_morphology_filter(morphology_filter)
            if self.logger.debug_level == 'full':
                axs[2+i].imshow(frame.img)
                axs[2+i].set_title(str(morphology_filter))
    
class HDABProcessor(Processor):
    """
    Subclass of Processor which handles the DAB/Hematoxylin (and Residual) stains
    :param logger: Logger object which will store various processing parameters
    :param frame: input RGB frame to process
    :param apply_smoothing: applies an anisotropic diffusion smoothing filter
    :param normalize_background: normalizes the background to a constant value for better thresholding
    :param stain_vector: overrides the default staining vector
    """
    def __init__(
            self,
            logger: sana.logging.Logger,
            frame: sana.image.Frame, 
            apply_smoothing: bool=True,
            normalize_background: bool=True,
            stain_vector: list=None,
            **kwargs
    ):
        super(HDABProcessor, self).__init__(logger, frame, **kwargs)

        # separate out the individual stains within the image
        self.ss = sana.color_deconvolution.StainSeparator('H-DAB', stain_vector)
        stains = self.ss.separate(self.frame.img)
        self.hem = sana.image.frame_like(self.frame, stains[:,:,0])
        self.dab = sana.image.frame_like(self.frame, stains[:,:,1])
        self.res = sana.image.frame_like(self.frame, stains[:,:,2])

        if logger.debug_level == 'full':
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(20,15))
            axs = axs.ravel()
            ax = axs[0]
            ax.imshow(self.frame.img)
            ax.set_title('Frame')
            ax = axs[1]
            ax.imshow(self.hem.img, cmap='gray')
            ax.set_title('HEM (OD)')
            ax = axs[2]
            ax.imshow(self.dab.img, cmap='gray')
            ax.set_title('DAB (OD)')
            ax = axs[3]
            ax.imshow(self.res.img, cmap='gray')
            ax.set_title('RES (OD)')

        # rescale the OD to uint8 using the digital min/max
        # TODO: maybe dont use digital min/max?
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        # smooth the DAB, mainly flattening interiors of objects
        if apply_smoothing:
            self.dab.anisodiff()

        # subtract the bacgkround image from the stains
        if normalize_background:
            self.dab.remove_background()

        self.logger.data['apply_smoothing'] = apply_smoothing
        self.logger.data['normalize_background'] = normalize_background
        self.logger.data['stain_vector'] = stain_vector
        
        if self.logger.debug_level == 'full':
            ax = axs[5]
            ax.imshow(self.dab.img, cmap='gray')
            ax.set_title('DAB (Preprocessed)')

    def run(self, triangular_strictness=0.0, minimum_threshold=0, od_threshold=None, morphology_filters=[], **kwargs):

        # list of processed images to return
        ret = {
            'main_mask': self.main_mask,
            'sub_masks': self.sub_masks,
            'exclusion_mask': self.exclusion_mask,
            'valid_mask': self.valid_mask,
        }

        # get the threshold for the DAB that is inside the main roi
        if od_threshold is None:
            hist = self.dab.get_histogram(mask=self.main_mask)
            threshold = sana.threshold.triangular_method(
                hist, 
                strictness=triangular_strictness,
                debug=self.logger.debug_level == 'full'
            )
            if threshold < minimum_threshold:
                threshold = minimum_threshold
                
        # manually select the threshold
        else:
            threshold = 255 * (od_threshold - self.ss.min_od[1]) / \
                (self.ss.max_od[1] - self.ss.min_od[1])
            
        # perform pixel classification using thresholding and morphology filters
        self.positive_dab = self.dab.copy()
        self.classify_pixels(self.positive_dab, threshold, morphology_filters=morphology_filters)
        ret['positive_dab'] = self.positive_dab

        # return all of the processed images
        return ret
