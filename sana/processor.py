
# installed modules
import numpy as np
from matplotlib import pyplot as plt

# sana modules
from sana.image import ImageTypeException, create_mask
from sana.features import Heatmap

class Processor:
    """
    Generic antibody processor class. This class sets ROI and mask attributes, and contains functions for generating quantitative results from a frame
    :param frame: Frame to process
    :param roi_type: type of ROI the Frame came from. If "GM", then cortical profiles will be generated
    :param qupath_threshold: manually selected threshold defined in qupath
    :param stain_vector: manually selected stain vector
    :param save_images: whether or not to write out intermediate images. WARNING: this may use up a lot of disk space
    :param run_wildcat: whether or not to run the pre-trained wildcat model on this frame, if the model exists
    :param run_cells: whether or not to run the cell detection algorithm
    """
    def __init__(self, frame, roi_type="", qupath_threshold=None, stain_vector=None, save_images=False, run_wildcat=False, run_cells=False):
        self.frame = frame
        self.roi_type = roi_type
        self.qupath_threshold = qupath_threshold
        self.stain_vector = stain_vector
        self.save_images = save_images
        self.run_wildcat = run_wildcat
        self.run_cells = run_cells

    def run(self, odir, params, main_roi, sub_rois=[], ignore_rois=[]):
        """
        This function generates the frame masks from the input ROIs. Subclass implementations of this function will handle the actual quantitation
        """

        # generate the main mask
        self.main_roi = main_roi
        self.main_mask = create_mask([self.main_roi], self.frame, x=0, y=255)

        # generate the sub masks
        self.sub_rois = []
        self.sub_masks = []
        for i in range(len(sub_rois)):
            if sub_rois[i] is None:
                self.sub_rois.append(None)
                self.sub_masks.append(None)
            else:
                self.sub_rois.append(sub_rois[i])
                self.sub_masks.append(create_mask([self.sub_rois[i]], self.frame, x=0, y=255))
        #
        # end of sub_masks loop

        # generate the ignore mask
        self.ignore_rois = ignore_rois
        self.ignore_mask = create_mask(self.ignore_rois, self.frame, x=255, y=0)

        # apply the ignore mask to the other masks
        self.main_mask.mask(self.ignore_mask)
        [sub_mask.mask(self.ignore_mask) for sub_mask in self.sub_masks if not sub_mask is None]

        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIGINAL')

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
        # NOTE: dividing by max value ensures this is a non-zero pixel count. This is probably faster than counting non-zero
        # TODO: not quite right, what if the whole mask is zeros
        main_area = np.sum(self.main_mask.img / np.max(self.main_mask.img))

        # get the positive classified area in the frame
        pos = np.sum(frame.img / np.max(frame.img))

        # calculate %AO of the main ROI
        main_ao = pos / main_area

        # DEBUG: show the original frame and the pixel classifications
        if frame.logger.debug:
            fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
            axs[0].imshow(frame.img)
            axs[1].imshow(frame.img)

        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = [], []
        for sub_mask in self.sub_masks:
            if sub_mask is None:
                sub_aos.append(np.nan)
                sub_areas.append(np.nan)
            else:
                tmp_frame = frame.copy()
                tmp_frame.mask(sub_mask)
                sub_area = np.sum(sub_mask.img / np.max(sub_mask.img))
                pos = np.sum(tmp_frame.img / np.max(tmp_frame.img))
                sub_ao = pos / sub_area
                sub_aos.append(sub_ao)
                sub_areas.append(sub_area)

        return main_ao, main_area, sub_ao, sub_areas
    
    def calculate_heatmaps(self, frame, tsize, tstep, detections=[]):
        """
        This function calculates various quantitative values during a convolution process. The result is a heatmap of each variable, and when the roi_type='GM', also outputs signal representing the features as a function of GM depth
        :param frame: input Frame, see calculate_ao() for more info
        :param tsize: tile size used during convolution
        :param tstep: tile step length used during convolution
        :param detections: list of Polygons detected in the Frame
        """

        # calculate the feature heatmaps (nfeats,H,W)
        # TODO: other features to generate add functions here!
        heatmap = Heatmap(frame, detections, tsize, tstep)
        featuremaps = heatmap.run([heatmap.ao, heatmap.density, heatmap.area])

        # special deformation and averaging done only on GM ROIs
        if self.roi_type == 'GM':

            # calculate the cortical profiles (1,nfeats,H)
            signals = np.mean(featuremaps, axis=2)[None,...]

            # deform the heatmap to the main mask, then calculate cortical profile (1,nfeats,nsamp)
            main_deformed_featuremaps = heatmap.deform(featuremaps, [self.main_mask])
            main_deformed_signals = np.mean(main_deformed_featuremaps, axis=3)

            # deform the heatmap to the sub masks and calculate profile (nmasks,nfeats,nsamp)
            if len(self.sub_masks) != 0:
                sub_deformed_featuremaps = heatmap.deform(featuremaps, self.sub_masks)
                sub_deformed_signals = np.mean(sub_deformed_featuremaps, axis=3)
            else:
                sub_deformed_featuremaps = None
                sub_deformed_signals = None
        else:
            signals = None
            main_deformed_featuremaps = None
            main_deformed_signals = None
            sub_deformed_featuremaps = None
            sub_deformed_signals = None

        return featuremaps, signals, main_deformed_featuremaps, main_deformed_signals, sub_deformed_featuremaps, sub_deformed_signals

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

        # apply the threshold
        frame.threshold(threshold, 0, 1)
        frame_thresh = frame.copy()

        # apply the closing filter
        frame.closing_filter(closing_radius)

        # apply the opening filter
        frame.opening_filter(opening_radius)

        # make sure the morphological filters did not add any extra positive pixels
        frame.img = ((frame.img != 0) & (frame_thresh.img != 0)).astype(np.uint8)

    
    def save_frame(self, frame, odir, suffix):
        """
        Writes the input frame to a PNG file
        """
        filename = io.create_filepath(filename=self.frame.filename, ext='.png', suffix=suffix, filepath=odir)
        frame.save(filename)

    def save_array(self, arr, odir, suffix):
        """
        Writes a numpy array to a .npy file
        """
        filename = io.create_filepath(filename=self.frame.filename, ext='.npy', suffix=suffix, filepath=odir)
        np.save(filename, arr)

    def save_parameters(self, odir, parameters):
        """
        Writes a Parameters object to a csv file
        """
        filename = io.create_filepath(filename=self.frame.filename, ext='.csv', filepath=odir)

    def save_binary_frame(self, frame, odir, suffix=""):
        """
        Writes a compressed binary image array to a .dat file. This is useful when performing a slidescan operation to save disk space
        """
