
# system modules
import os
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

# installed modules
import numpy as np
import cv2
import skimage.feature
import skfmm
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# sana modules
import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.color_deconvolution
import pdnl_sana.image
import pdnl_sana.threshold
import pdnl_sana.geo
import pdnl_sana.slide

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
            main_roi: sana.geo.Polygon=None,
            sub_rois: [sana.geo.Polygon]=[],
            exclusion_rois: [sana.geo.Polygon]=[],
            main_mask: sana.image.Frame=None,
    ):
        self.logger = logger
        self.frame = frame

        # generate the main mask
        if not main_mask is None:
            self.main_mask = main_mask
        else:
            if main_roi is None:
                self.main_mask = sana.image.frame_like(self.frame, np.ones(self.frame.shape[:2], dtype=np.uint8))
            else:
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
            axs[0].imshow(frame.img, cmap='gray')
            axs[0].set_title('Original')

        # apply the threshold and the mask
        frame.threshold(threshold)
        if not mask is None:
            frame.mask(mask)

        # DEBUG:
        if self.logger.debug_level == 'full':
            axs[1].imshow(frame.img, cmap='gray')
            axs[1].set_title('Threshold')

        # apply each filter
        for i, morphology_filter in enumerate(morphology_filters):
            frame.apply_morphology_filter(morphology_filter)
            if self.logger.debug_level == 'full':
                axs[2+i].imshow(frame.img, cmap='gray')
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
            radius: int=100,
            stain_vector: list=None,
            run_hem=True,
            run_dab=True,
            **kwargs
    ):
        super(HDABProcessor, self).__init__(logger, frame, **kwargs)

        # separate out the individual stains within the image
        self.ss = sana.color_deconvolution.StainSeparator('H-DAB', stain_vector)
        #self.ss.estimate_stain_vector(self.frame.img)
        stains = self.ss.separate(self.frame.img)
        self.hem = sana.image.frame_like(self.frame, stains[:,:,0])
        self.dab = sana.image.frame_like(self.frame, stains[:,:,1])
        self.res = sana.image.frame_like(self.frame, stains[:,:,2])
        self.stains = [self.hem, self.dab, self.res]

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
        # TODO: this compresses the digital space, maybe don't use min/max od!
        self.hem.img = self.hem.img - self.dab.img
        if run_hem:
            self.hem.rescale(self.ss.min_od[0], self.ss.max_od[1])
        if run_dab:
            self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        # smooth the DAB, mainly flattening interiors of objects
        if apply_smoothing:
            if run_hem:
                self.hem.anisodiff()
            if run_dab:
                self.dab.anisodiff()

        # subtract the bacgkround image from the stains
        if normalize_background:
            if run_hem:
                self.hem.remove_background(radius=radius)
            if run_dab:
                self.dab.remove_background(radius=radius)

        self.logger.data['apply_smoothing'] = apply_smoothing
        self.logger.data['normalize_background'] = normalize_background
        self.logger.data['stain_vector'] = stain_vector
        
        if self.logger.debug_level == 'full':
            ax = axs[4]
            ax.imshow(self.hem.img, cmap='gray')
            ax.set_title('HEM (Preprocessed)')
            ax = axs[5]
            ax.imshow(self.dab.img, cmap='gray')
            ax.set_title('DAB (Preprocessed)')

    def run(self, triangular_strictness=0.0, minimum_threshold=0, od_threshold=None, mask=None, morphology_filters=[], target_stain="DAB"):

        # list of processed images to return
        ret = {
            'main_mask': self.main_mask,
            'sub_masks': self.sub_masks,
            'exclusion_mask': self.exclusion_mask,
            'valid_mask': self.valid_mask,
        }
        stain_idx = self.ss.stain_vector.stains.index(target_stain)
        stain = self.stains[stain_idx]

        # get the threshold for the stain using pixels that are inside the ROI
        if od_threshold is None:
            hist = stain.get_histogram(mask=self.main_mask)
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
        positive_stain = stain.copy()
        self.classify_pixels(positive_stain, threshold, mask=mask, morphology_filters=morphology_filters)
        ret['stain'] = stain
        ret['positive_stain'] = positive_stain

        self.logger.data['triangular_strictness'] = triangular_strictness
        self.logger.data['minimum_threshold'] = minimum_threshold
        self.logger.data['od_threshold'] = od_threshold
        self.logger.data['morphology_filters'] = morphology_filters
        self.logger.data['threshold'] = threshold

        # return all of the processed images
        return ret

def preprocess_wsi_chunk_wrapper(args):
    return preprocess_wsi_chunk(*args)
def preprocess_wsi_chunk(temp_dir, i, j, slide_f, frame_size, level, rois):
    """
    This function loads, preprocesses, and caches a chunk of a WSI
    """
    logger = sana.logging.Logger('normal', os.path.join(temp_dir, f'parameters_{i}_{j}.pkl'))
    loader = sana.slide.Loader(logger, slide_f)
    frame_size = sana.geo.Point(frame_size, frame_size, is_micron=False, level=level)
    for roi in rois:
        roi.is_micron = False
        roi.level = level
    framer = sana.slide.Framer(loader, size=frame_size, step=frame_size, level=level, rois=rois)

    # get the frame mask
    mask = framer.load_mask(i, j)

    # decide if it's worth it to process this frame
    if np.sum(mask.img) < 0.005*mask.img.shape[0]*mask.img.shape[1]:
        return None, None

    # extract the frame from the WSI
    frame = framer.load_frame(i, j)

    # preprocess the frame
    processor = HDABProcessor(
        logger, frame, main_mask=mask, 
        run_hem=True, run_dab=True, 
        apply_smoothing=False, 
        normalize_background=True, radius=100
    )

    # cache the stain data
    processor.hem.save(os.path.join(temp_dir, f"hem_{i}_{j}.png"))
    processor.dab.save(os.path.join(temp_dir, f"dab_{i}_{j}.png"))
    mask.save(os.path.join(temp_dir, f"mask_{i}_{j}.png"))
    logger.write_data()

    # return the histograms in order to calculate a global WSI
    hem_histogram = processor.hem.get_histogram(mask=mask)
    dab_histogram = processor.dab.get_histogram(mask=mask)

    return hem_histogram, dab_histogram
    

