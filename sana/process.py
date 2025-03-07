
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
import sana.sta
import sana.filter

class Processor:
    """
    Generic antibody processor class. This class sets ROI and mask attributes, and contains functions for generating quantitative results from a frame
    :param logger: sana.logging.Logger
    :param frame: sana.image.Frame object to process
    :param stain_vector: manually selected stain vector to be used during color deconvolution
    """
    def __init__(self, logger, frame, stain_vector=None):
        self.logger = logger
        self.frame = frame
        self.stain_vector = stain_vector

        # TODO: stain vector is unimplemented!

    def run(self, main_roi, sub_rois=[], ignore_rois=[], **kwargs):
        """
        This function generates the frame masks from the input ROIs. Subclass implementations of this function will handle the actual quantitation
        """

        # generate the main mask
        self.main_roi = main_roi.to_polygon()
        self.main_mask = sana.image.create_mask([self.main_roi], self.frame, x=0, y=1)

        # generate the sub masks
        self.sub_rois = []
        self.sub_masks = []
        for i in range(len(sub_rois)):
            if sub_rois[i] is None:
                self.sub_rois.append(None)
                self.sub_masks.append(None)
            else:
                self.sub_rois.append(sub_rois[i].to_polygon())
                self.sub_masks.append(sana.image.create_mask([self.sub_rois[i]], self.frame, x=0, y=1))
        #
        # end of sub_masks loop

        # generate the ignore mask
        self.ignore_rois = [ignore_roi.to_polygon() for ignore_roi in ignore_rois]
        self.ignore_mask = sana.image.create_mask(self.ignore_rois, self.frame, x=0, y=1)
        self.keep_mask = sana.image.create_mask(self.ignore_rois, self.frame, x=1, y=0)

        # apply the ignore mask to the other masks
        self.main_mask.mask(self.keep_mask)
        [sub_mask.mask(self.keep_mask) for sub_mask in self.sub_masks if not sub_mask is None]

    def plot_roi(self, ax):
        ax.plot(*self.main_roi.T, color='black')
        [ax.plot(*x.T, color='gray') for x in self.sub_rois if not x is None]
        [ax.plot(*x.T, color='black') for x in self.ignore_rois]

    def classify_pixels(self, frame, threshold, mask=None, morphology_filters=[], debug=True):
        """
        Classifies pixels as foreground/background based on a threshold and morphological filtering
        :param img: input Frame
        :param threshold: threshold for input Frame
        :param mask: mask to apply to the input
        :param closing_radius: radius of morphological closing kernel
        :param opening_radius: radius of morphological opening kernel
        """

        # apply the mask
        if not mask is None:
            frame.mask(mask)

        # DEBUG:
        if self.logger.generate_plots and debug:
            fig, axs = plt.subplots(1, 2+len(morphology_filters), sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            axs[0].set_title('Original')

        # apply the threshold
        frame.threshold(threshold, 0, 1)

        # DEBUG:
        if self.logger.generate_plots and debug:
            axs[1].imshow(frame.img)
            axs[1].set_title('Threshold')

        # apply each filter
        for i, morphology_filter in enumerate(morphology_filters):
            frame.apply_morphology_filter(morphology_filter)

            # DEBUG:
            if self.logger.generate_plots and debug:
                axs[2+i].imshow(frame.img)
                axs[2+i].set_title(str(morphology_filter))
    
    def detect_somas(self, stain, pos, minimum_soma_radius=1):

        # apply the distance transform on the positive pixels convert centers of objects into peaks
        img_proc = cv2.distanceTransform(pos.img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # min-max filter iterations are used to further help separate images
        img_proc = sana.filter.min_max_filter(
            img_proc,
            r=minimum_soma_radius, 
            n_iterations=1,
        )

        # set the peak distance to be between 1 radius and 1 diameter
        # NOTE: accounts for overlapping somas and ellipsoid objects
        min_distance = int(round(1.5*minimum_soma_radius))

        # set the peak threshold to be pixel is greater than 75% of nearby pixels
        threshold_abs = 0.75

        # 2d peak detection to find the centers of somas
        soma_ctrs = skimage.feature.peak_local_max(
            img_proc,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
        )

        # inverse the (x,y) points to (y,x) for future processing
        soma_ctrs = soma_ctrs[:,::-1]

        if self.logger.generate_plots:
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(stain.img)
            axs[0].set_title('Original Stain')
            axs[1].imshow(pos.img)
            axs[1].set_title('Classified Pixels')
            axs[2].imshow(img_proc)
            axs[2].plot(*soma_ctrs.T, '*', color='red')
            axs[2].set_title('Soma Peaks')
            axs[3].imshow(self.frame.img)
            axs[3].plot(*soma_ctrs.T, '*', color='red')
            axs[3].set_title('Frame w/ Soma Detections')

        return soma_ctrs
    
    def segment_somas(self, pos, soma_ctrs, n_directions, sigma_x, sigma_y, fm_threshold):
        
        # define the directions to test
        thetas = np.linspace(0, np.pi, n_directions)
        filters = [sana.filter.AnisotropicGaussianFilter(th=th, sg_x=sigma_x, sg_y=sigma_y) for th in thetas]
        if self.logger.generate_plots:
            fig, axs = plt.subplots(4,int(np.ceil(len(filters)/4)))
            axs = axs.ravel()
            for i, f in enumerate(filters):
                axs[i].imshow(f.kernel, cmap='gray')
            fig.suptitle('Rotated Anisotropic Gaussian Kernels')

        # apply the filters to the image
        self.logger.debug('Applying anisotropic filters...')
        directions = np.zeros((n_directions, pos.size()[1], pos.size()[0]), dtype=float)
        for k in tqdm(range(n_directions)):
            directions[k] = filters[k].apply(pos.img[:,:,0])

        # calculate the directional ratio of the image
        mi = np.min(directions, axis=0)
        mx = np.max(directions, axis=0)
        directional_ratio = np.divide(mi**3, mx, where=mx!=0)

        # apply fast march algorithm using soma centers as the starting points
        phi = np.full_like(pos.img[:,:,0], -1, dtype=int)
        phi[soma_ctrs[:,1], soma_ctrs[:,0]] = 1

        # speed of the march is based on the directional ratio algorithm
        speed = directional_ratio
        speed[pos.img[:,:,0] == 0] = 0

        # run the fast march
        t = skfmm.travel_time(phi, speed)
        t[t.mask] = np.inf

        # threshold the fast march
        soma_mask = sana.image.frame_like(pos, (t < fm_threshold).astype(np.uint8))

        # perform instance segmentation to separate nearby somas
        # soma_polygons = soma_mask.instance_segment(self.soma_ctrs)
        soma_polygons = soma_mask.to_polygons()

        if self.logger.generate_plots:
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            axs = axs.ravel()
            ax = axs[0]
            ax.imshow(self.frame.img)
            ax.plot(*soma_ctrs.T, '*', color='red')
            ax.set_title('Frame')
            ax = axs[1]
            ax.imshow(speed, cmap='coolwarm')
            ax.plot(*soma_ctrs.T, '*', color='red')
            ax.set_title('Directional Ratio')
            ax = axs[2]
            ax.imshow(t, cmap='coolwarm', vmax=fm_threshold*2.0, vmin=0)
            [ax.plot(*x.T, color='red', linewidth=1) for x in soma_polygons]
            ax.set_title('Fast March Algorithm')
            ax = axs[3]
            ax.imshow(self.frame.img)
            [ax.plot(*x.T, color='red') for x in soma_polygons]
            ax.set_title('Soma Segmentations')


        return soma_polygons

class HDABProcessor(Processor):
    """
    Subclass of Processor which handles the DAB and Hematoxylin (and Residual) stains, providing functions to detect positive inclusions in DAB/HEM and detect objects from the positive pixels
    """
    def __init__(self, logger, frame, 
                 run_smoothing=True,
                 run_background_subtraction=True, 
                 subtract_dab=True,
                 **kwargs):
        super(HDABProcessor, self).__init__(logger, frame, **kwargs)
        self.run_smoothing = run_smoothing
        self.run_background_subtraction = run_background_subtraction
        self.subtract_dab = subtract_dab

        self.ss = sana.color_deconvolution.StainSeparator('H-DAB', self.stain_vector)
        self.stains = self.ss.separate(self.frame.img)
        self.hem = sana.image.frame_like(self.frame, self.stains[:,:,0])
        self.dab = sana.image.frame_like(self.frame, self.stains[:,:,1])
        self.res = sana.image.frame_like(self.frame, self.stains[:,:,2])

        if logger.generate_plots:
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
            axs = axs.ravel()
            ax = axs[0]
            ax.imshow(self.frame.img)
            ax.set_title('Frame')
            ax = axs[1]
            ax.imshow(self.hem.img)
            ax.set_title('HEM (OD)')
            ax = axs[2]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (OD)')
            ax = axs[3]
            ax.imshow(self.res.img)
            ax.set_title('RES (OD)')

        # TODO: clip low DAB by some value to avoid deleting faint neurons. Really we just want to remove the super dark DAB (i.e. pial surface or strong inclusions that aren't actually cells)
        if self.subtract_dab:
            self.hem.img -= self.dab.img

        # rescale the OD to uint8
        # TODO: maybe dont use digital min/max?
        self.hem.rescale(0.0, 1.0)
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        # smooth the stains, mainly flattening interiors of objects
        # NOTE: this is always good for hematoxylin
        self.hem.anisodiff()
        if self.run_smoothing:
            self.dab.anisodiff()

        # subtract the bacgkround image from the stains
        # NOTE: this is always good for hematoxylin
        # self.hem.remove_background()
        if self.run_background_subtraction:
            self.dab.remove_background()

        if self.logger.generate_plots:
            ax = axs[4]
            ax.imshow(self.hem.img)
            ax.set_title('Hematoxylin (Preprocessed)')
            ax = axs[5]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (Preprocessed)')

    def run(self,
            triangular_strictness=0.0,
            minimum_threshold=0,
            od_threshold=None,
            morphology_filters=[],
            run_soma_detection=False,
            minimum_soma_radius=1,
            min_max_filter_iterations=1,
            run_soma_segmentation=False,
            n_directions=12,
            fm_threshold=10,
            directional_sigma=5,
            run_object_segmentation=False,
            use_watershed=False,
            run_sta=False,
            sta_sigma=(10,10),
            sta_strictness=0.0,
            run_model=False,
            **kwargs):
        super().run(**kwargs)

        # list of processed images to return
        ret = []

        # get the threshold for the DAB that inside the main roi
        if od_threshold is None:
            hist = self.dab.get_histogram(mask=self.main_mask)
            threshold = sana.threshold.triangular_method(
                hist, 
                strictness=triangular_strictness,
                debug=self.logger.generate_plots
            )
            if threshold < minimum_threshold:
                threshold = minimum_threshold
        else:
            threshold = 255 * (od_threshold - self.ss.min_od[1]) / \
                (self.ss.max_od[1] - self.ss.min_od[1])
            
        # perform pixel classification by thresholding and morphology filters
        self.thr_dab = self.dab.copy()
        self.classify_pixels(self.thr_dab, threshold, mask=self.keep_mask, debug=False)
        self.pos_dab = self.dab.copy()
        self.classify_pixels(self.pos_dab, threshold, mask=self.keep_mask, morphology_filters=morphology_filters)
        ret.append(['pos_dab', self.pos_dab])

        # run the structure tensor analysis to find the orientation of the DAB
        # TODO: move to function
        if run_sta:
            coh, ang = sana.sta.run_sta(self.dab, sta_sigma)
            
            # # get the coherence of the image, and the "strength" of each direction
            # coh, ang = sana.sta.run_directional_sta(self.dab, sta_sigma)
            
            # # generate a probability image using coh, DAB, and angular strength
            # prob = coh * ang * self.dab.img.astype(float)
            # prob /= np.max(prob)

            # angular_thresholds = []
            # for i in range(prob.shape[2]):
            #     p = prob[:,:,i].flatten()
            #     hist = np.histogram(p, 255)[0].astype(float)
            #     angular_thresholds.append(sana.threshold.triangular_method(hist, strictness=sta_strictness) / 255)

            # V = sana.image.Frame(((prob[:,:,0] >= angular_thresholds[0]) & (np.argmax(prob, axis=2) == 0)).astype(np.uint8))
            # H = sana.image.Frame(((prob[:,:,1] >= angular_thresholds[1]) & (np.argmax(prob, axis=2) == 1)).astype(np.uint8))
            # D = sana.image.Frame(((prob[:,:,2] >= angular_thresholds[2]) & (np.argmax(prob, axis=2) == 2)).astype(np.uint8))
            # ret.append(['vertical', V])
            # ret.append(['horizontal', H])
            # ret.append(['diagonal', D])
            ret.append(['coh', coh*self.dab.img.astype(float)])
            ret.append(['ang', ang])

            if self.logger.generate_plots:
                overlay = self.frame.copy()
                overlay = sana.image.overlay_mask(overlay, V, color='red')
                overlay = sana.image.overlay_mask(overlay, H, color='blue')
                overlay = sana.image.overlay_mask(overlay, D, color='orange')
                fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
                axs[0].imshow(self.frame.img)
                axs[1].imshow(prob)
                axs[2].imshow(overlay.img)

        if run_soma_detection:

            # find the centers of each soma using distance transform and peak detection
            self.logger.debug('Finding centers of somas...')
            self.soma_ctrs = self.detect_somas(self.dab, self.pos_dab, minimum_soma_radius)
            ret.append(['soma_ctrs', self.soma_ctrs])

        if run_soma_segmentation:
            if not hasattr(self, 'soma_ctrs'):
                raise ProcessingException("Must run Soma Detection before Soma Segmentation")

            # segment the somas using directional radius, fast march, watershed, and soma centers
            self.logger.debug('Segmenting somas...')
            self.soma_polygons = self.segment_somas(self.dab, self.soma_ctrs, n_directions=n_directions, sigma_x=directional_sigma/3, sigma_y=directional_sigma, fm_threshold=fm_threshold)
            ret.append(['soma_polygons', self.soma_polygons])

        if run_object_segmentation:
            if not hasattr(self, 'soma_ctrs'):
                raise ProcessingException("Must run Soma Detection before Object Segmentation")
            
            if use_watershed:

                # segment the objects using watershed and the soma centers
                self.object_polygons = self.thr_dab.instance_segment(self.soma_ctrs, debug=self.logger.generate_plots)
            
            else:
                polygons = self.thr_dab.to_polygons()

                # filter the polygons by the soma centers
                self.object_polygons = []
                for (x,y) in self.soma_ctrs:
                    for i in range(len(polygons)):
                        if sana.geo.ray_tracing(x, y, polygons[i]):
                            self.object_polygons.append(polygons[i])
                            break
                    else:
                        continue
                    # polygons.pop(i)

            ret.append(['object_polygons', self.object_polygons])

            if self.logger.generate_plots:
                fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
                ax.imshow(self.frame.img)
                [ax.plot(*x.T, color='red') for x in self.object_polygons]
                if hasattr(self, "soma_polygons"):
                    [ax.plot(*x.T, color='blue') for x in self.soma_polygons]
                else:
                    [ax.plot(*self.soma_ctrs.T, '*', color='blue')]
                
                ax.set_title('Frame w/ Object Segmentations')
                self.plot_roi(ax)

            # TODO: add skeletonization and process connection

        # run the specified machine learning model
        if run_model:

            # TODO: if microglia, 
            # TODO: these functions will store PYR/GRAN/Tangle specific AOs
            pass

        # return all of the processed images
        return ret
    
class ProcessingException(Exception):
    def __init__(self, message):
        self.message = message
