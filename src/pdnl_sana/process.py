
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
from numba import jit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# sana modules
import pdnl_sana.color_deconvolution
import pdnl_sana.image
import pdnl_sana.threshold
import pdnl_sana.geo
import pdnl_sana as sana
import pdnl_sana.logging
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
            main_mask: sana.image.Frame=None
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
    
def segment_wsi_chunk_wrapper(args):
    return segment_wsi_chunk(*args)
def segment_wsi_chunk(temp_dir, i, j, hem_threshold, dab_threshold):
    if not os.path.exists(os.path.join(temp_dir, f"hem_{i}_{j}.png")):
        return

    logger = sana.logging.Logger('normal', os.path.join(temp_dir, f'parameters_{i}_{j}.pkl'))
    frame_loc = logger.data['loc']

    hem = sana.image.Frame(os.path.join(temp_dir, f"hem_{i}_{j}.png"))
    dab = sana.image.Frame(os.path.join(temp_dir, f"dab_{i}_{j}.png"))
    mask = sana.image.Frame(os.path.join(temp_dir, f"mask_{i}_{j}.png"))
    hem_int = hem.copy()

    # apply the global thresholds
    hem.threshold(hem_threshold)
    dab.threshold(dab_threshold)

    # clean up the stains
    dab.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    hem.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    hem.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', 2))

    # remove positive DAB and pixels outside the mask
    hem.mask(dab, invert=True)
    hem.mask(mask)

    # find all the somas throughout the counterstain
    # TODO: parameter should be in microns!
    soma_ctrs = sana.process.detect_somas(hem, minimum_soma_radius=3)
    
    # segment the somas using polygons
    soma_polygons, _ = hem.instance_segment(soma_ctrs)[0]
    soma_polygons = [p for p in soma_polygons if not type(p) is list and len(p) != 0]
    
    # move the polygons into the slide coordinate system
    [p.translate(-frame_loc) for p in soma_polygons]

    # re-calculate the centers of the polygons using the bounding box
    soma_bbs = [p.bounding_box() for p in soma_polygons]
    soma_ctrs = np.array([loc + size//2 for (loc, size) in soma_bbs])

    # feature 1: calculate the area of each polygon
    soma_areas = np.array([p.get_area() for p in soma_polygons])

    # feature 2: calculate the mean HEM intensity within the polygon
    soma_ints = []
    for poly in soma_polygons:

        # move to frame coordinate system
        poly.translate(frame_loc)
        
        # extract tile based on the bounding box of the polygon
        loc, size = poly.bounding_box()
        tile = sana.image.Frame(hem_int.get_tile(loc, size))

        # create a mask of pixels within the polygon
        poly.translate(loc)
        tile_mask = sana.image.create_mask_like(tile, [poly])
        poly.translate(-loc)

        # calculate average intensity
        soma_ints.append(np.mean(tile.img[tile_mask.img != 0]))

        # move back to slide coordinate system
        poly.translate(-frame_loc)

    # combine into an (N,4) array
    soma_feats = np.concatenate([
        soma_ctrs, 
        soma_areas[:,None], 
        np.array(soma_ints)[:,None]], 
        axis=1)

    # cache the cell features
    np.save(os.path.join(temp_dir, f"feats_{i}_{j}.npy"), soma_feats)


def train_wm_segmenter(heatmap, wm_coords=None, gm_coords=None, priors=None):

    heatmap = heatmap.copy()
    h, w, d = heatmap.img.shape    
    feats = heatmap.img.reshape(h*w, d)
    non_zero = ~np.any(feats == 0, axis=1)

    ss = StandardScaler()
    ss.fit(feats[non_zero])
    feats = ss.transform(feats)

    # fig, axs = plt.subplots(1,d)
    # for i in range(d):
    #     axs[i].hist(feats[non_zero,i], bins=100, density=True,
    #             color='gray', alpha=0.7, edgecolor='k')

    model = Pipeline([
        ('ss', ss),
        ('gmm', GaussianMixture(n_components=2, covariance_type='full')),
    ])

    if not wm_coords is None:
        wm_coords = np.ravel_multi_index(wm_coords.T, (h,w))
        gm_coords = np.ravel_multi_index(gm_coords.T, (h,w))
        means_init = np.array([
            np.mean(feats[wm_coords], axis=0),
            np.mean(feats[gm_coords], axis=0),
        ])
        print(means_init)
        model.set_params(gmm__means_init=means_init)

        
        # for i in range(d):
        #     axs[i].axvline(means_init[0,i], color='red')
        #     axs[i].axvline(means_init[1,i], color='blue')

        model.get_params()['gmm'].fit(feats[non_zero])
        model.wm_label = 0
        
    elif not priors is None:
        model.get_params()['gmm'].fit(feats[non_zero])

        mu = model.get_params()['gmm'].means_

        # store votes for each label
        votes = np.zeros(mu.shape[0], dtype=int)
    
        # check each feature prior
        for i in range(priors.shape[0]):
    
            # figure out which label has the most extreme value for this feature
            if priors[i] == 1:
                label = np.argmax(mu[:,i])
            else:
                label = np.argmin(mu[:,i])
    
            # vote for this label
            votes[label] += 1
        model.wm_label = np.argmax(votes)
        model.label_agrees_with_priors = np.max(votes) == len(priors)
    else:
        print('need information on which label is wm')
        return None

    return model

def deploy_wm_segmenter(model, heatmap, open_r=21, close_r=2):
    h, w, d = heatmap.img.shape
    pred = model.predict(heatmap.img.reshape(h*w, d)).reshape(h, w, 1)
    
    wm_mask = sana.image.frame_like(heatmap, (pred == model.wm_label).astype(np.uint8))
    wm_mask.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', close_r))
    wm_mask.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', open_r))

    return wm_mask
