
# installed modules
import numpy as np
import cv2
from tqdm import tqdm

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, overlay_thresh
from sana_heatmap import Heatmap
from sana_geo import Point
from sana_filters import minmax_filter

# debugging modules
from matplotlib import pyplot as plt

TSIZE = Point(400, 100, is_micron=False, lvl=0)
TSTEP = Point(50, 50, is_micron=False, lvl=0)

# generic processor class, sets the main attributes and holds
# functions for generating data from processed Frames
class Processor:
    def __init__(self, fname, frame, logger, roi_type="", qupath_threshold=None, stain_vector=None):
        self.fname = fname
        self.frame = frame
        self.logger = logger
        self.roi_type = roi_type
        self.qupath_threshold = qupath_threshold
        self.stain_vector = stain_vector
    #
    # end of constructor

    def generate_masks(self, main_roi, sub_rois=[]):
        # generate the main mask
        self.main_mask = create_mask(
            [main_roi],
            self.frame.size(), self.frame.lvl, self.frame.converter,
            x=0, y=255, holes=[]
        )

        # generate the sub masks
        self.sub_masks = []
        for i in range(len(sub_rois)):
            if sub_rois[i] is None:
                self.sub_masks.append(None)
            else:
                mask = create_mask(
                    [sub_rois[i]],
                    self.frame.size(), self.frame.lvl, self.frame.converter,
                    x=0, y=255, holes=[]
                )
                self.sub_masks.append(mask)
        #
        # end of sub_masks loop
    #
    # end of gen_masks

    # generic function to calculate %AO of a thresholded frame
    # 1) gets the %AO of the main_roi
    # 2) gets the %AO of the sub_rois provided (if any)
    #
    # NOTE: the frame and rois must be in the same coord. system
    # TODO: any way to check the above??, could check if theres no overlap between frame nad mask
    #       IMPORTANT! not actually AO=0, if the frame is black you would still get AO=0
    def run_ao(self, frame, detections=[]):

        # apply the mask
        frame.mask(self.main_mask)

        # get the total area of the roi
        # TODO: don't just divide by 255, make sure to check first!!
        area = np.sum(self.main_mask.img / 255)

        # get the pos. area in the frame
        pos = np.sum(frame.img / 255)

        # calculate %AO of the main roi
        ao = pos / area

        # apply the sub masks and get the %AO of each
        sub_aos, sub_areas = [], []
        for sub_mask in self.sub_masks:
            if sub_mask is None:
                sub_aos.append(np.nan)
                sub_areas.append(np.nan)
            else:
                tmp_frame = frame.copy()
                tmp_frame.mask(sub_mask)
                sub_area = np.sum(sub_mask.img / 255)
                sub_pos = np.sum(tmp_frame.img / 255)
                sub_ao = sub_pos / sub_area
                sub_aos.append(sub_ao)
                sub_areas.append(sub_area)

        # calculate feature signals as a function of depth
        if 'GM' in self.roi_type:
            signals = self.get_signals(frame, detections)
        else:
            signals = None

        # finally, return the results
        ret = {
            'ao': ao, 'area': area,
            'sub_aos': sub_aos, 'sub_areas': sub_areas,
            'signals': signals,
        }
        return ret
    #
    # end of run_ao

    def segment_cells(self, frame, threshold,
                      disk_r, sigma, n_iterations, close_r, open_r, clean_r):

        # threshold the image and filter lots of data out to just get large circular objects
        img_objs = self.get_thresh(frame.img, threshold, self.main_mask, close_r, clean_r)

        # run the distance transforms to find parts of circular objects far away from background
        # TODO: this normalize is a little dangerous, think about blank images
        # TODO: don't really need to normalize, minmax handles all that since its relateive to the disk
        img_dist = cv2.distanceTransform(img_objs, cv2.DIST_L2, 3)
        img_dist = cv2.normalize(img_dist, 0, 255, cv2.NORM_MINMAX)

        # run the minmax filter to find the centers of the cells
        img_minmax = minmax_filter(img_dist, disk_r, sigma, n_iterations)

        # create sure foregournd using the minimas of the minmax image
        # NOTE: the centers will be -1's, a little unintuitive
        candidates = np.where(img_minmax == -1)
        r0 = 3
        sure_fg = np.zeros_like(frame.img, dtype=np.uint8)[:,:,0]
        for i in range(len(candidates[0])):
            sure_fg = cv2.circle(sure_fg, (candidates[1][i], candidates[0][i]),
                                 r0, color=255, thickness=-1)

        # threshold the frame for all cell parts using smaller kernel for segmentation
        img_thresh = self.get_thresh(frame.img, threshold, self.main_mask, close_r, open_r)

        # create sure background using multiple dilations
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        sure_bg = cv2.dilate(img_thresh, dil_kern, iterations=3)

        # create unknown pixel regions, whatever is left from sure_fg and sure_bg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # run the watershed algorithm to perform instance segmentation using the cell centers
        # TODO: try gauss blur before watershed?
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        img_thresh_rgb = np.stack((img_thresh,)*3, axis=-1)
        markers = cv2.watershed(img_thresh_rgb, markers)
        markers[markers <= 1] = 0

        # generate Polygons from the instance segmented markers image
        cells = []
        z = np.zeros((markers.shape[0], markers.shape[1]), np.uint8)
        o = z.copy() + 1
        for val in tqdm(range(1, np.max(markers)+1)):
            x = np.where(markers == val, o, z)
            f = Frame(x, frame.lvl, frame.converter)
            f.get_contours()
            f.filter_contours()
            bodies = f.get_body_contours()
            if len(bodies) != 0:
                cells.append(bodies[0].polygon.connect())

        if self.logger.plots:
            # plot all the intermediate images
            fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(frame.img)
            axs[0].set_title('Orig DAB')
            axs[4].matshow(img_objs, cmap='gray')
            axs[4].set_title('Proc. DAB for Objs only')
            axs[1].imshow(img_dist, cmap='gray')
            axs[1].set_title('Distance Transform')
            axs[5].imshow(img_minmax, cmap='gray')
            axs[5].set_title('Min-Max Filtered Dist. Transform')
            axs[5].plot(candidates[1], candidates[0], 'x', color='red')
            axs[2].matshow(img_thresh, cmap='gray')
            axs[2].set_title('Proc. DAB for All Cells Parts')
            axs[6].matshow(255-sure_bg, cmap='gray')
            axs[6].set_title('Sure Background Data')
            axs[3].matshow(unknown, cmap='gray')
            axs[3].set_title('Unknown Pixel Data')
            axs[7].matshow(markers, cmap='rainbow')
            axs[7].set_title('Instance Segmented Neurons')
            
            fig.tight_layout()
        #
        # end of debugging plots
        
        return cells
    #
    # end of segment_cells

    def get_thresh(self, img, threshold, mask, close_r, open_r):
        img = img.copy()
        img[mask.img == 0] = 0
        img_thresh = np.where(img < threshold, 0, 255).astype(np.uint8)[:,:,0]
        close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r, close_r))
        open_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_r, open_r))
        img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, close_kern)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, open_kern)
        img_final = 255 * ((img_open != 0) & (img_thresh != 0)).astype(np.uint8)
        return img_final
    #
    # end of get_thresh
    
    def get_signals(self, frame, detections=[], tsize=None, tstep=None):

        # these are the size and step length of the convolution process
        if tsize is None:
            tsize = TSIZE
        if tstep is None:
            tstep = TSTEP

        # calculate the feature heatmaps
        heatmap = Heatmap(frame, detections, tsize, tstep)
        feats = heatmap.run([heatmap.ao, heatmap.density, heatmap.area])

        # deform the heatmap to the main and sub mask
        # in: 3, H, W
        # out: 4, 3, H, W
        main_deformed_feats = heatmap.deform(feats, [self.main_mask])
        if len(self.sub_masks) != 0:
            sub_deformed_feats = heatmap.deform(feats, self.sub_masks)

        # calculate the signals as the average over the columns
        signals = {
            'normal': np.mean(feats, axis=2)[None,...],
            'main_deform': np.mean(main_deformed_feats, axis=3),
        }

        if len(self.sub_masks) != 0:
            signals['sub_deform'] = np.mean(sub_deformed_feats, axis=3)

        # print(feats.shape, frame.img.shape, main_deformed_feats.shape, sub_deformed_feats.shape)
        # fig, axs = plt.subplots(2,4)
        # axs[0][0].imshow(frame.img)
        # axs[0][1].imshow(feats[0])
        # axs[0][2].imshow(main_deformed_feats[0][0])
        # axs[1][0].imshow(sub_deformed_feats[0][0])
        # axs[1][1].imshow(sub_deformed_feats[1][0])
        # axs[1][2].imshow(sub_deformed_feats[2][0])
        # axs[1][3].imshow(sub_deformed_feats[3][0])

        # fig, axs = plt.subplots(1,3)
        # axs[0].plot(signals['normal'][0][0])
        # axs[1].plot(signals['main_deform'][0][0])
        # for i in range(signals['sub_deform'].shape[0]):
        #     n = signals['sub_deform'].shape[2]
        #     x = np.arange(n) + i*n
        #     axs[2].plot(x, signals['sub_deform'][i][0])

        # plt.show()

        return signals
    #
    # end of get_signals

    def save_frame(self, odir, frame, suffix):
        fpath = sana_io.create_filepath(
            self.fname, ext='.png', suffix=suffix, fpath=odir)
        frame.save(fpath)
    #
    # end of save_frame

    # NOTE: signals is a [NSignals, NFeats, NSamples] Array
    def save_signals(self, odir, signals, suffix):
        fpath = sana_io.create_filepath(
            self.fname, ext='.npy', suffix=suffix, fpath=odir)
        np.save(fpath, signals)
    #
    # end of save_curve

    def save_array(self, odir, arr, suffix):
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        frame = Frame(np.rint(255*arr).astype(np.uint8))
        self.save_frame(odir, frame, suffix)
    #
    # end of save_array

    def save_params(self, odir, params):
        fpath = sana_io.create_filepath(self.fname, ext='.csv', fpath=odir)
        params.write_data(fpath)
    #
    # end of save_params
#
# end of Processor

#
# end of file
