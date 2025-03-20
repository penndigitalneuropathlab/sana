
# system modules
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

# installed modules
import numpy as np
import cv2
import skimage.feature
import skfmm
from matplotlib import pyplot as plt
from numba import jit

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
        ret['positive_stain'] = positive_stain

        # return all of the processed images
        return ret

def detect_somas(pos, minimum_soma_radius=1):
    dist = cv2.distanceTransform(pos.img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    ctrs = skimage.feature.peak_local_max(
        dist,
        min_distance=int(round(1.5*minimum_soma_radius)), # accounts for overlapping/elliptical somas
    )[:,::-1]
    
    return ctrs

def segment_somas(pos, ctrs, n_directions=2, stride=1, sigma=3, fm_threshold=10, npools=1, instance_segment=True):

    if len(ctrs) == 0:
        return []
    
    # create the rotated anisotropic gaussian filters
    step = sana.geo.Point(stride, stride, False, pos.level)
    thetas = np.linspace(0, np.pi, n_directions, endpoint=False)
    filters = [sana.filter.AnisotropicGaussianFilter(th=th, sg_x=1, sg_y=sigma) for th in thetas]

    # apply the filters to the positive pixels
    def apply_filter(filt, frame, step):
        return filt.apply(frame, step)
    if npools == 1:
        directions = [apply_filter(filt, pos, step) for filt in filters]
    else:
        directions = ThreadPool(npools).map(partial(apply_filter, frame=pos, step=step), filters)

    # calculate directional ratio using the min and max directional values
    mi = np.min(directions, axis=0)
    mx = np.max(directions, axis=0)
    directional_ratio = cv2.resize(np.divide(mi**3, mx, where=mx!=0), dsize=(pos.img.shape[1], pos.img.shape[1]), interpolation=cv2.INTER_NEAREST)

    # initial points for fast march algorithm are the soma centers
    phi = np.full_like(pos.img[:,:,0], -1, dtype=int)
    phi[ctrs[:,1], ctrs[:,0]] = 1

    # speed is derived from the directional ratio
    speed = directional_ratio
    speed[pos.img[:,:,0] == 0] = 0

    # run the fast march
    time = skfmm.travel_time(phi, speed)
    time[time.mask] = np.inf

    # threshold the time to create the soma mask
    mask = sana.image.frame_like(pos, (time < fm_threshold).astype(np.uint8))
    
    # convert the mask to polygons
    if instance_segment:
        polygons = mask.instance_segment(ctrs)
    else:
        polygons = mask.to_polygons()

    return polygons
