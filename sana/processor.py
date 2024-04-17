
# installed modules
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
from copy import copy
from multiprocessing import Manager, Process

# sana modules
from sana.image import ImageTypeException, create_mask, frame_like, Frame
from sana.color_deconvolution import StainSeparator

from sana_thresholds import max_dev
from sana_filters import minmax_filter

class Processor:
    """
    Generic antibody processor class. This class sets ROI and mask attributes, and contains functions for generating quantitative results from a frame
    :param logger: sana.logging.Logger
    :param frame: sana.image.Frame object to process
    :param qupath_threshold: manually selected threshold defined in qupath
    :param stain_vector: manually selected stain vector to be used during color deconvolution
    """
    def __init__(self, logger, frame, qupath_threshold=None, stain_vector=None):
        self.logger = logger
        self.frame = frame
        self.qupath_threshold = qupath_threshold
        self.stain_vector = stain_vector

        # TODO: qupath threshold and stain vector are unimplemented!

    def run(self, main_roi, sub_rois=[], ignore_rois=[], **kwargs):
        """
        This function generates the frame masks from the input ROIs. Subclass implementations of this function will handle the actual quantitation
        """

        # generate the main mask
        self.main_roi = main_roi.to_polygon()
        self.main_mask = create_mask([self.main_roi], self.frame, x=0, y=1)

        # generate the sub masks
        self.sub_rois = []
        self.sub_masks = []
        for i in range(len(sub_rois)):
            if sub_rois[i] is None:
                self.sub_rois.append(None)
                self.sub_masks.append(None)
            else:
                self.sub_rois.append(sub_rois[i].to_polygon())
                self.sub_masks.append(create_mask([self.sub_rois[i]], self.frame, x=0, y=1))
        #
        # end of sub_masks loop

        # generate the ignore mask
        self.ignore_rois = [ignore_roi.to_polygon() for ignore_roi in ignore_rois]
        self.ignore_mask = create_mask(self.ignore_rois, self.frame, x=1, y=0)

        # apply the ignore mask to the other masks
        self.main_mask.mask(self.ignore_mask)
        [sub_mask.mask(self.ignore_mask) for sub_mask in self.sub_masks if not sub_mask is None]

    def calculate_ao(self, frame):
        """
        Calculates the %Area Occupied of the input frame. This is the percentage of positively classified pixels in the image. It is calculated over the provided main and sub ROIs.
        :param frame: input frame, not necessarily self.frame, since multiple Frames could be generated from self.frame and various %AO values can be calculated
        """
        if not frame.is_binary():
            raise ImageTypeException("Input frame must be a binary image for %AO quantitation")

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

    def classify_pixels(self, frame, threshold, mask=None, closing_radius=0, opening_radius=0):
        """
        Classifies pixels as foreground/background based on a threshold and morphological filtering
        :param img: input Frame
        :param threshold: threshold for input Frame
        :param mask: mask Frame to apply to the input
        :param closing_radius: radius of morphological closing kernel
        :param opening_radius: radius of morphological opening kernel
        """

        # apply the mask
        if not mask is None:
            frame.mask(mask)

        # DEBUG:
        if self.logger.generate_plots:
            fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            axs[0].set_title('Original')

        # apply the threshold
        frame_thresh = frame.copy()
        frame_thresh.threshold(threshold, 0, 1)

        # DEBUG:
        if self.logger.generate_plots:
            axs[1].imshow(frame_thresh.img)
            axs[1].set_title('Threshold')

        # apply the closing filter
        frame_filt = frame_thresh.copy()
        if closing_radius > 0:
            frame_filt.closing_filter(closing_radius)

            # DEBUG:
            if self.logger.generate_plots:
                axs[2].imshow(frame_filt.img)
                axs[2].set_title('Morphological Closing')

        # apply the opening filter
        if opening_radius > 0:
            frame_filt.opening_filter(opening_radius)

            # DEBUG:
            if self.logger.generate_plots:
                axs[3].imshow(frame_filt.img)
                axs[3].set_title('Morphological Opening')

        # make sure the morphological filters did not add any extra positive pixels
        frame.img = ((frame_thresh.img == 1) & (frame_filt.img == 1)).astype(np.uint8)

        # DEBUG:
        if self.logger.generate_plots:
            axs[4].imshow(frame.img)
            axs[4].set_title('Final')

        return frame
    
class DABProcessor(Processor):
    """
    Subclass of Processor which handles the DAB stain, providing functions to detect positive DAB
    """
    def __init__(self, logger, frame, **kwargs):
        super(DABProcessor, self).__init__(logger, frame, **kwargs)

        if self.logger.generate_plots:
            fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
            axs = axs.ravel()
            ax = axs[0]
            ax.imshow(self.frame.img)
            ax.set_title('Original Frame')

        self.ss = StainSeparator('H-DAB', self.stain_vector)
        self.stains = self.ss.separate(self.frame.img)
        self.hem = frame_like(self.frame, self.stains[:,:,0])
        self.dab = frame_like(self.frame, self.stains[:,:,1])
        if logger.generate_plots:
            ax = axs[1]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (OD)')

        # TODO: potentially subtract HEM

        # rescale the OD DAB image to uint8
        # TODO: maybe dont use digital min/max?
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])
        if logger.generate_plots:
            ax = axs[2]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (Rescaled)')

        # smooth the DAB
        self.dab.anisodiff()
        if logger.generate_plots:
            ax = axs[3]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (Smoothed)')

        # background removal
        self.dab.remove_background()
        if logger.generate_plots:
            ax = axs[4]
            ax.imshow(self.dab.img)
            ax.set_title('DAB (Background Subtracted)')

    def run(self,
            max_dev_scaler=1.0,
            max_dev_endpoint=255,
            opening_radius=0,
            closing_radius=0, 
            **kwargs):
        super().run(**kwargs)

        # get the threshold for the DAB
        hist = self.dab.get_histogram()
        threshold = max_dev(hist, scale=max_dev_scaler, mx=max_dev_endpoint, debug=self.logger.generate_plots)

        self.pos_dab = self.classify_pixels(self.dab, threshold, self.main_mask, closing_radius=closing_radius, opening_radius=opening_radius)

        main_ao, main_area, sub_aos, sub_areas = self.calculate_ao(self.pos_dab)
        self.logger.data['area'] = main_area
        self.logger.data['ao'] = main_ao
        self.logger.data['sub_areas'] = sub_areas
        self.logger.data['sub_aos'] = sub_aos

        if self.logger.generate_plots:
            plt.show()

class HematoxylinProcessor(Processor):
    """
    Subclass of Processor which handles the hematoxylin stain, providing functions to detect cells in the hematoxylin.
    """
    def __init__(self, logger, frame, **kwargs):
        super(HematoxylinProcessor, self).__init__(logger, frame, **kwargs)
        if self.logger.generate_plots:
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            ax = axs[0][0]
            ax.imshow(self.frame.img)
            ax.set_title('Original Frame')

        # get the HEM image
        self.ss = StainSeparator('H-DAB', self.stain_vector)
        self.stains = self.ss.separate(self.frame.img)
        self.hem = frame_like(self.frame, self.stains[:,:,0])
        self.dab = frame_like(self.frame, self.stains[:,:,1])
        if self.logger.generate_plots:
            ax = axs[0][1]
            ax.imshow(self.hem.img)
            ax.set_title('Hematoxylin (OD)')

        # TODO: clip low DAB by some value to avoid deleting faint neurons. Really we just want to remove the super dark DAB (i.e. pial surface or strong inclusions that aren't actually cells)
        self.hem.img -= self.dab.img
        if self.logger.generate_plots:
            ax = axs[1][0]
            ax.imshow(self.hem.img)
            ax.set_title('HEM w/ DAB subtracted (OD)')

        # rescale the OD HEM image to uint8
        # NOTE: using the digital min/max heavily compresses the range, setting a constant for now should be good enough
        self.hem.rescale(0.0, 1.0)
        # self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        if self.logger.generate_plots:
            ax = axs[1][1]
            ax.imshow(self.hem.img, cmap='gray')
            ax.set_title('Hematoxylin (Rescaled)')
            hist = self.hem.get_histogram()

        # smooth the interiors of the cells
        # NOTE: this helps when we represent cells as disks
        self.hem.anisodiff()

        # self.disk_radius = 5 # anything smaller and we detect everything
        # self.sigma = 5 # further smoothing
        # self.closing_radius = 5 # set this higher for dendrites in NeuN
        # self.opening_radius = 7
        # self.cleaning_radius = 11 # removes things like vessels

        self.disk_radius = 5 # anything smaller and we detect everything
        self.sigma = 5 # further smoothing
        self.closing_radius = 7 # set this higher for dendrites in NeuN
        self.opening_radius = 0
        self.cleaning_radius = 0 # removes things like vessels

        # TODO: this is bad! need to implement csf threshold detection from thumbnail, then convert that to HEM space
        csf_threshold = 10

        # calculate the histogram, and remove CSF data
        hist = self.hem.get_histogram()
        hist[:csf_threshold] = 0

        # TODO: analyze these params!
        scale = 1.0
        mx = 100
        # TODO: rewrite this!
        self.threshold = max_dev(hist, scale=scale, mx=mx, debug=self.logger.generate_plots)

    def segment_cells(self):

        # apply threshold and filter out most of the data to find large circular objects
        object_pixels = self.hem.copy()
        self.classify_pixels(object_pixels, self.threshold, None, closing_radius=self.closing_radius, opening_radius=self.opening_radius)

        # find parts of circular objects that are far away from background pixels
        img_dist = cv2.distanceTransform(object_pixels.img, cv2.DIST_L2, 3)

        img_minmax = minmax_filter(img_dist, self.disk_radius, self.sigma, 1)

        candidates = np.where(img_minmax == -1)
        r0 = 3
        sure_fg = np.zeros_like(self.frame.img, dtype=np.uint8)
        for i in range(len(candidates[0])):
            sure_fg = cv2.circle(
                sure_fg, 
                (candidates[1][i], candidates[0][i]), 
                r0, color=255, thickness=-1)

        cell_pixels = self.hem.copy()
        self.classify_pixels(cell_pixels, self.threshold, None, closing_radius=self.closing_radius, opening_radius=self.opening_radius)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        sure_bg = cv2.dilate(cell_pixels.img, kern, iterations=3)

        unknown_pixels = cv2.subtract(sure_bg, sure_fg[:,:,0])

        ret, markers = cv2.connectedComponents(sure_fg[:,:,0])
        markers += 1
        markers[unknown_pixels == 1] = 0
        hem_rgb = np.concatenate([self.hem.img, self.hem.img, self.hem.img], axis=-1) # TODO: hem or orig
        markers = cv2.watershed(hem_rgb, markers)
        markers[markers <= 1] = 0

        njobs = 8
        cells = self.markers_to_cells(markers, njobs)

        # return cells

        rgb_markers = np.zeros_like(self.frame.img)
        colors = [(np.random.randint(10, 255),
                    np.random.randint(10, 255),
                    np.random.randint(10, 255)) \
                    for _ in range(1,np.max(markers))]
        colors = [(0,0,0)] + colors + [(80,80,80)]
        for j in tqdm(range(rgb_markers.shape[0])):
            for i in range(rgb_markers.shape[1]):
                rgb_markers[j,i] = colors[markers[j,i]]

        fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(self.frame.img)
        axs[0].set_title('Original Frame')
        axs[1].matshow(object_pixels.img, cmap='gray')
        axs[1].set_title('Proc. Stain for Objs only')
        axs[2].imshow(img_dist, cmap='gray')
        axs[2].set_title('Distance Transform')
        axs[3].matshow(cell_pixels.img, cmap='gray')
        axs[3].set_title('Proc. Stain for All Cells Parts')
        axs[4].imshow(self.hem.img)
        axs[4].set_title('Original Stain')
        axs[5].matshow(unknown_pixels, cmap='gray')
        axs[5].set_title('Unknown Pixel Data')                        
        axs[6].imshow(img_minmax, cmap='gray')
        axs[6].set_title('Min-Max Filtered Dist. Transform')
        axs[6].plot(candidates[1], candidates[0], 'x', color='red')
        axs[7].matshow(markers, cmap='rainbow')
        axs[7].set_title('Instance Segmented Cells')
        fig.tight_layout()

        return cells

    def markers_to_cells(self, markers, njobs):
        
        # get the the height of each section of the frame to split into
        step = int(self.frame.size()[1]/njobs)
        args_list = []
        for n in range(njobs):
            st = n*step
            en = st+step
            args_list.append(
                {
                    'markers': markers[st:en, :].copy(), # TODO: is this copy necessary?
                    'level': self.frame.level,
                    'converter': copy(self.frame.converter),
                    'st': st,
                    'en': en,
                }
            )
        manager = Manager()
        ret = manager.dict()
        jobs = []
        for pid in range(njobs):
            p = Process(target=run_segment_markers,
                        args=(pid, ret, args_list[pid]))
            jobs.append(p)
        [p.start() for p in jobs]
        [p.join() for p in jobs]

        cells = []
        for pid in ret:
            cells += ret[pid]
        cells = [x for x in cells if not x is None]

        # TODO: this is related to the issue in sana_process where pickling loses these attributes
        for cell in cells:
            cell.is_micron = False
            cell.level = self.frame.level
            cell.order = 1
         
        return cells
    #
    # end of markers_to_cells
        

def run_segment_markers(pid, ret, args):
    ret[pid] = segment_markers(**args)
    
def segment_markers(markers, level, converter, st, en):
    cells = []
    z = np.zeros((markers.shape[0], markers.shape[1]), np.uint8)
    o = z.copy() + 1
    marker_vals = np.sort(np.unique(markers))[1:]
    for val in marker_vals:
        x = np.where(markers == val, o, z)
        f = Frame(x, level, converter)
        bodies, holes = f.get_contours() # TODO: specify area limits
        if len(bodies) != 0:
            cell = bodies[0].polygon.connect()
            cell[:,1] += st
            cells.append(cell)
    return cells

#
# end of file
