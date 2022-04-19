
# installed modules
import numpy as np
import cv2
from tqdm import tqdm

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, overlay_thresh, create_mask
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_antibody_processor import Processor
from sana_filters import minmax_filter

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

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

    # performs a simple threshold using a manually selected cut off point
    # then runs the %AO process
    def run_manual_ao(self, odir, params):

        # apply the thresholding
        self.manual_dab_thresh = self.dab.copy()
        self.manual_dab_thresh.threshold(self.manual_dab_threshold, 0, 255)

        results = self.run_ao(self.manual_dab_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']
        params.data['manual_ao'] = results['ao']
        params.data['manual_sub_aos'] = results['sub_aos']
        params.data['manual_stain_threshold'] = self.manual_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir, 'manual_ao')

        # save the images used in processing
        self.manual_overlay = overlay_thresh(
            self.frame, self.manual_dab_thresh)        
        self.save_frame(odir, self.manual_dab_thresh, 'THRESH')
        self.save_frame(odir, self.manual_overlay, 'QC')        
    #
    # end of run_manual_ao

    # performs normalization, smoothing, and histogram
    # TODO: rename scale to something better
    # TODO: add switches to turn off/on mean_norm, anisodiff, morph
    def run_auto_ao(self, odir, params, scale=1.0):
        
        # normalize the image
        self.dab_norm = mean_normalize(self.dab)

        # smooth the image
        self.dab_norm.anisodiff()

        # get the histograms
        self.dab_hist = self.dab.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        # get the stain threshold
        self.auto_dab_threshold = max_dev(self.dab_hist, scale=scale)
        self.auto_dab_norm_threshold = max_dev(self.dab_norm_hist, scale=scale)

        # apply the thresholding
        self.auto_dab_norm_thresh = self.dab_norm.copy()
        self.auto_dab_norm_thresh.threshold(self.auto_dab_norm_threshold, 0, 255)
        
        # run the AO process
        results = self.run_ao(self.auto_dab_norm_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']        
        params.data['auto_ao'] = results['ao']
        params.data['auto_sub_aos'] = results['sub_aos']
        params.data['auto_stain_threshold'] = self.auto_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir, 'auto_ao')

        # save the images used in processing
        self.auto_overlay = overlay_thresh(
            self.frame, self.auto_dab_norm_thresh)        
        self.save_frame(odir, self.dab_norm, 'PROB')
        self.save_frame(odir, self.auto_dab_norm_thresh, 'THRESH')
        self.save_frame(odir, self.auto_overlay, 'QC')
    #
    # end of run_auto_ao

    # TODO: comment and describe parameters, why current values are selected!
    def run_segment(self, odir, params,
                disk_r=7, sigma=3,
                close_r=9, open_r=5, debug=False):

        n_iterations = 2
        hem_cells = self.detect_hem_cells(
            params, disk_r, n_iterations, sigma, close_r, open_r, debug)

    def detect_hem_cells(self, params,
                         disk_r, sigma, n_iterations, close_r, open_r, debug):

        # get the rescale parameters
        hist = self.hem.histogram()        
        vmi = np.argmax(hist)
        vmx = 92

        # # rescale the image
        # self.hem.img = (self.hem.img.astype(float) - mi) / (mx-mi)
        # self.hem.img = np.clip(self.hem.img, 0, None)
        # self.hem.img = (255 * self.hem.img).astype(np.uint8)
        
        # smooth the image
        self.hem.anisodiff()

        self.hem_hist = self.hem.histogram()
        self.hem_threshold = max_dev(self.hem_hist, mx=vmx)
        
        # get the image array
        img = self.hem.img.copy()[:,:,0]
        
        # do some thresholding and morph filters to
        # 1) remove faint objects (ambiguous)
        # 2) delete tiny objects (too small/fragments)
        # 3) close holes in center of faint neurons
        thresh_img = img.copy()
        thresh_img = np.where(thresh_img < self.hem_threshold, 0, 255)
        thresh_img = thresh_img.astype(np.uint8)
        close_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_r, close_r))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, close_kern)
        open_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_r, open_r))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, open_kern)

        # mask out all unwanted data as described above
        img[thresh_img == 0] = 0

        # define the sure background, anything 0 and not close in prox. to data
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        sure_bg = cv2.dilate(thresh_img, dil_kern, iterations=3)

        # perform the min-max filtering to maximize centers of cells
        # TODO: inverse this filter
        img_minmax = minmax_filter(255-img, disk_r, sigma, n_iterations, debug)

        # define the sure foreground, small circles at minimums of filter output
        candidates = np.where((img_minmax == -1) & (img != 0))
        r0 = 3
        sure_fg = np.zeros_like(img, dtype=np.uint8)
        for i in range(len(candidates[0])):
            sure_fg = cv2.circle(sure_fg,
                                 (candidates[1][i], candidates[0][i]),
                                 r0, color=255, thickness=-1)

        # get the unknown areas of the image, not defined by sure bg or fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # run the watershed algo
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        rgb_hem = np.stack((self.hem.img[:,:,0],)*3, axis=-1)
        markers = cv2.watershed(rgb_hem, markers)
        rgb_markers = np.zeros_like(rgb_hem)
        colors = [(np.random.randint(10, 255),
                   np.random.randint(10, 255),
                   np.random.randint(10, 255)) \
                  for _ in range(1,np.max(markers))]
        colors = [(0,0,0),(0,0,0)] + colors + [(80,80,80)]
        for j in tqdm(range(rgb_markers.shape[0])):
            for i in range(rgb_markers.shape[1]):
                rgb_markers[j,i] = colors[markers[j,i]]
                
        # mframe = self.frame.copy()
        # mframe.img = np.where(markers <= 0, 0, 1).astype(np.uint8)
        # mframe.get_contours()
        # mframe.filter_contours(min_body_area=8)
        # markers = create_mask(
        #     [c.polygon for c in mframe.contours],
        #     self.frame.size(), self.frame.lvl, self.frame.converter, 0, 255)
        
        fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
        axs[0].imshow(self.frame.img)
        axs[0].set_title('orig')
        axs[1].imshow(img, vmin=vmi, vmax=vmx)
        axs[1].set_title('preprocess')
        axs[2].imshow(img_minmax)
        axs[2].plot(candidates[1], candidates[0], '+', color='red', markersize=3)
        axs[2].set_title('min-max filter')
        axs[3].imshow(rgb_markers)
        axs[3].set_title('segmented cells')
        axs[4].imshow(self.frame.img)
        axs[4].plot(candidates[1], candidates[0], '+', color='black', markersize=4)
        axs[4].set_title('candidate locs')
        plt.show()
#
# end of HDABProcessor

#
# end of file
