
# installed modules
import numpy as np
from matplotlib import pyplot as plt
import cv2
import skfmm
from tqdm import tqdm
from shapely import geometry

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
        self.ignore_mask = sana.image.create_mask(self.ignore_rois, self.frame, x=1, y=0)

        # apply the ignore mask to the other masks
        self.main_mask.mask(self.ignore_mask)
        [sub_mask.mask(self.ignore_mask) for sub_mask in self.sub_masks if not sub_mask is None]

    def calculate_ao(self, frame):
        """
        Calculates the %Area Occupied of the input frame. This is the percentage of positively classified pixels in the image. It is calculated over the provided main and sub ROIs.
        :param frame: input frame, not necessarily self.frame, since multiple Frames could be generated from self.frame and various %AO values can be calculated
        """
        if not frame.is_binary():
            raise sana.image.ImageTypeException("Input frame must be a binary image for %AO quantitation")

        # apply the mask
        frame.mask(self.main_mask)

        # get the total area of the roi
        main_area = np.sum(self.main_mask.img)

        # get the positive classified area in the frame
        pos = np.sum(frame.img)

        # calculate %AO of the main ROI
        main_ao = pos / main_area

        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = [], []
        for sub_mask in self.sub_masks:
            if sub_mask is None:
                sub_aos.append(np.nan)
                sub_areas.append(np.nan)
            else:
                tmp_frame = frame.copy()
                tmp_frame.mask(sub_mask)
                sub_area = np.sum(sub_mask.img)
                pos = np.sum(tmp_frame.img)
                sub_ao = pos / sub_area
                sub_aos.append(sub_ao)
                sub_areas.append(sub_area)

        return main_ao, main_area, sub_aos, sub_areas
    
    # TODO: refactor to fit Convolver
    # def calculate_heatmaps(self, frame, tsize, tstep, detections=[]):
    #     """
    #     This function calculates various quantitative values during a convolution process. The result is a heatmap of each variable, and when the roi_type='GM', also outputs signal representing the features as a function of GM depth
    #     :param frame: input Frame, see calculate_ao() for more info
    #     :param tsize: tile size used during convolution
    #     :param tstep: tile step length used during convolution
    #     :param detections: list of Polygons detected in the Frame
    #     """

    #     # calculate the feature heatmaps (nfeats,H,W)
    #     # TODO: other features to generate add functions here!
    #     heatmap = Heatmap(frame, detections, tsize, tstep)
    #     featuremaps = heatmap.run([heatmap.ao, heatmap.density, heatmap.area])

    #     # special deformation and averaging done only on GM ROIs
    #     if self.roi_type == 'GM':

    #         # calculate the cortical profiles (1,nfeats,H)
    #         signals = np.mean(featuremaps, axis=2)[None,...]

    #         # deform the heatmap to the main mask, then calculate cortical profile (1,nfeats,nsamp)
    #         main_deformed_featuremaps = heatmap.deform(featuremaps, [self.main_mask])
    #         main_deformed_signals = np.mean(main_deformed_featuremaps, axis=3)

    #         # deform the heatmap to the sub masks and calculate profile (nmasks,nfeats,nsamp)
    #         if len(self.sub_masks) != 0:
    #             sub_deformed_featuremaps = heatmap.deform(featuremaps, self.sub_masks)
    #             sub_deformed_signals = np.mean(sub_deformed_featuremaps, axis=3)
    #         else:
    #             sub_deformed_featuremaps = None
    #             sub_deformed_signals = None
    #     else:
    #         signals = None
    #         main_deformed_featuremaps = None
    #         main_deformed_signals = None
    #         sub_deformed_featuremaps = None
    #         sub_deformed_signals = None

    #     return featuremaps, signals, main_deformed_featuremaps, main_deformed_signals, sub_deformed_featuremaps, sub_deformed_signals

    def classify_pixels(self, frame, threshold, mask=None, morphology_filters=[]):
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
        if self.logger.generate_plots:
            fig, axs = plt.subplots(1, 2+len(morphology_filters), sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            axs[0].set_title('Original')

        # apply the threshold
        frame.threshold(threshold, 0, 1)

        # DEBUG:
        if self.logger.generate_plots:
            axs[1].imshow(frame.img)
            axs[1].set_title('Threshold')

        # apply each filter
        for i, morphology_filter in enumerate(morphology_filters):
            frame.apply_morphology_filter(morphology_filter)

            # DEBUG:
            if self.logger.generate_plots:
                axs[2+i].imshow(frame.img)
                axs[2+i].set_title(str(morphology_filter))
    
    def detect_somas(self, pos, minimum_soma_radius=1, min_max_filter_iterations=1):

        # apply the distance transform
        img_dist = cv2.distanceTransform(pos.img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # apply min-max filter to find centers of somas
        if min_max_filter_iterations > 0:
            img_minmax = sana.filter.min_max_filter(self.frame, img_dist, minimum_soma_radius, min_max_filter_iterations, debug=self.logger.generate_plots)

            # get the coordinates of the soma centers
            soma_ctrs = np.array(np.where(img_minmax == 1)).T[:,::-1]
        else:
            soma_blobs = [x.polygon for x in pos.get_contours()[0]]
            soma_ctrs = np.rint(np.array([blob.get_centroid()[0] for blob in soma_blobs])).astype(int)

        if self.logger.generate_plots:
            fig, ax = plt.subplots(1,1)
            ax.imshow(self.frame.img)
            ax.plot(*soma_ctrs.T, '*', color='red')

        return soma_ctrs
    
    def segment_somas(self, pos, soma_ctrs, n_directions, sigma_x, sigma_y, fm_threshold):
        
        # define the directions to test
        thetas = np.linspace(0, np.pi, n_directions)
        filters = [sana.filter.AnisotropicGaussianFilter(th=th, sg_x=sigma_x, sg_y=sigma_y) for th in thetas]

        # apply the filters to the image
        self.logger.debug('Applying anisotropic filters...')
        directions = np.zeros((n_directions, pos.size()[1], pos.size()[0]), dtype=float)
        for k in tqdm(range(n_directions)):
            directions[k] = filters[k].apply(pos.img[:,:,0])

        # calculate the directional ratio of the image
        mi = np.min(directions, axis=0)
        mx = np.max(directions, axis=0)
        directional_ratio = np.divide(mi**3, mx, where=mx!=0)

        # apply fast march algorithm using soma centers as the 
        phi = np.full_like(pos.img[:,:,0], -1, dtype=int)
        phi[soma_ctrs[:,1], soma_ctrs[:,0]] = 1

        speed = directional_ratio
        speed[pos.img[:,:,0] == 0] = 0

        t = skfmm.travel_time(phi, speed)
        t[t.mask] = np.inf

        soma_mask = sana.image.frame_like(pos, (t < fm_threshold).astype(np.uint8))

        if self.logger.generate_plots:
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            axs = axs.ravel()
            ax = axs[0]
            ax.imshow(self.frame.img)
            ax.plot(*soma_ctrs.T, '*', color='red')
            ax = axs[1]
            ax.imshow(speed, cmap='gray')
            ax.plot(*soma_ctrs.T, '*', color='red')
            ax = axs[2]
            ax.imshow(t, cmap='gray', vmax=fm_threshold*1.5, vmin=0)
            overlay = sana.image.overlay_mask(self.frame, soma_mask)
            ax = axs[3]
            ax.imshow(overlay.img)

        return soma_mask

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
            ax.set_title('Original Frame')
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
        self.hem.remove_background()
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
            morphology_filters=[],
            run_soma_detection=False,
            minimum_soma_radius=1,
            min_max_filter_iterations=1,
            run_soma_segmentation=False,
            n_directions=12,
            fm_threshold=10,
            directional_sigma=5,
            run_object_segmentation=False,
            run_sta=False,
            sta_sigma=(10,10),
            sta_strictness=0.0,
            run_model=False,
            **kwargs):
        super().run(**kwargs)

        # list of processed images to return
        ret = []

        # get the threshold for the DAB
        hist = self.dab.get_histogram()
        threshold = sana.threshold.triangular_method(
            hist, 
            strictness=triangular_strictness,
            debug=self.logger.generate_plots
        )

        # perform pixel classification by thresholding and morphology filters
        self.pos_dab = self.dab.copy()
        self.classify_pixels(self.pos_dab, threshold, mask=self.main_mask, morphology_filters=morphology_filters)
        ret.append(['pos_dab', self.pos_dab])

        # run the structure tensor analysis to find the orientation of the DAB
        # TODO: move to function
        if run_sta:
            # get the coherence of the image, and the "strength" of each direction
            coh, ang = sana.sta.run_directional_sta(self.dab, sta_sigma)
            
            # generate a probability image using coh, DAB, and angular strength
            prob = coh * ang * self.dab.img.astype(float)
            prob /= np.max(prob)

            angular_thresholds = []
            for i in range(prob.shape[2]):
                p = prob[:,:,i].flatten()
                hist = np.histogram(p, 255)[0].astype(float)
                angular_thresholds.append(sana.threshold.triangular_method(hist, strictness=sta_strictness) / 255)

            V = sana.image.Frame(((prob[:,:,0] >= angular_thresholds[0]) & (np.argmax(prob, axis=2) == 0)).astype(np.uint8))
            H = sana.image.Frame(((prob[:,:,1] >= angular_thresholds[1]) & (np.argmax(prob, axis=2) == 1)).astype(np.uint8))
            D = sana.image.Frame(((prob[:,:,2] >= angular_thresholds[2]) & (np.argmax(prob, axis=2) == 2)).astype(np.uint8))
            ret.append(['vertical', V])
            ret.append(['horizontal', H])
            ret.append(['diagonal', D])

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

            # find the centers of each soma using min-max filter
            self.logger.debug('Finding centers of somas...')
            self.soma_ctrs = self.detect_somas(self.pos_dab, minimum_soma_radius, min_max_filter_iterations)
            ret.append(['soma_ctrs', self.soma_ctrs])

            # TODO: move to function
            if run_soma_segmentation:

                # segment the somas using directional radius and fast march
                # NOTE: using only threshold here no morphology filters
                self.logger.debug('Segmenting somas...')
                pos_pix = self.dab.copy()
                self.classify_pixels(pos_pix, threshold, mask=self.main_mask)
                self.soma_mask = self.segment_somas(pos_pix, self.soma_ctrs, n_directions=n_directions, sigma_x=directional_sigma/3, sigma_y=directional_sigma, fm_threshold=fm_threshold)
                ret.append(['soma_mask', self.soma_mask])

                # TODO: fast march combines nearby somas, probably need to watershed using the result of fast march

            # TODO: move to function
            if run_object_segmentation:

                sure_fg = np.zeros_like(self.pos_dab.img)[:,:,0]
                sure_fg[self.soma_ctrs[:,1], self.soma_ctrs[:,0]] = 1
                _, markers = cv2.connectedComponents(sure_fg)

                # watershed based on the already thresholded image
                pos_tiled = np.tile(self.pos_dab.img, 3)
                markers = cv2.watershed(pos_tiled, markers)

                # remove the background segmentation
                markers += 1
                bg = np.argmax(np.bincount(markers.flatten()))
                markers[markers == bg] = 0

                # convert to binary image
                binary_markers = markers.copy()
                binary_markers[binary_markers != 0] = 1
                binary_markers = binary_markers.astype(np.uint8)

                # contour detection to convert markers to polygons
                objs = []
                c, h = cv2.findContours(binary_markers, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                # look for body contours
                # TODO: this doesn't split if exactly 1 touching diagonal connection
                # TODO: looks like there are some objects missing?
                for i in range(len(c)):
                    if h[0][i][3] == -1:

                        # create the body polygon
                        body = geometry.Polygon(np.squeeze(c[i]))

                        # look for holes in this body
                        for j in range(len(c)):
                            if h[0][j][3] == i:

                                # create the hole polygon
                                hole = geometry.Polygon(np.squeeze(c[j]))

                                # remove the hole from the body
                                body = body.difference(hole)

                        res = sana.geo.from_shapely(body)
                        if type(res) is list:
                            objs += [x for x in res if not x is None]
                        elif not res is None:
                            objs.append(res)

                if self.logger.generate_plots:
                    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
                    axs[0].imshow(self.frame.img)
                    axs[1].imshow(markers)
                    axs[2].imshow(self.frame.img)
                    [axs[2].plot(*obj.T) for obj in objs]

                # TODO: add process connection

                # TODO: append object mask
                
                # TODO: append polygon list

        # run the specified machine learning model
        if run_model:

            # TODO: if microglia, 
            # TODO: these functions will store PYR/GRAN/Tangle specific AOs
            pass

        # return all of the processed images
        return ret