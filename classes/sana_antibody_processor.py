
# system modules
import io
from copy import copy
from multiprocessing import Manager, Process

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
    def __init__(self, fname, frame, logger, roi_type="", qupath_threshold=None,
                 save_images=False, run_wildcat=False, run_cells=False,
                 stain_vector=None):
        self.fname = fname
        self.frame = frame
        self.logger = logger
        self.roi_type = roi_type
        self.qupath_threshold = qupath_threshold
        self.save_images = save_images
        self.run_wildcat = run_wildcat
        self.run_cells = run_cells
        self.stain_vector = stain_vector
    #
    # end of constructor

    def generate_masks(self, main_roi, sub_rois=[]):
        # generate the main mask
        self.main_roi = main_roi
        self.main_mask = create_mask(
            [main_roi],
            self.frame.size(), self.frame.lvl, self.frame.converter,
            x=0, y=255, holes=[]
        )

        # generate the sub masks
        self.sub_rois = []
        self.sub_masks = []
        for i in range(len(sub_rois)):
            if sub_rois[i] is None:
                self.sub_rois.append(None)
                self.sub_masks.append(None)
            else:
                mask = create_mask(
                    [sub_rois[i]],
                    self.frame.size(), self.frame.lvl, self.frame.converter,
                    x=0, y=255, holes=[]
                )
                self.sub_rois.append(sub_rois[i])
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
        area = np.sum(self.main_mask.img / np.max(self.main_mask.img))

        # get the pos. area in the frame
        pos = np.sum(frame.img / np.max(frame.img))

        # calculate %AO of the main roi
        ao = pos / area

        if self.logger.plots:
            fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
            axs[0].imshow(self.frame.img)
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
                sub_pos = np.sum(tmp_frame.img / np.max(tmp_frame.img))
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
        img_dist = cv2.distanceTransform(img_objs, cv2.DIST_L2, 3)

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
        hem_rgb = np.concatenate([frame.img, frame.img, frame.img], axis=-1) # HOTFIX: hem or orig?
        markers = cv2.watershed(hem_rgb, markers)
        markers[markers <= 1] = 0

        njobs = 8
        cells = self.markers_to_cells(markers, njobs)

        if self.logger.plots:
            self.logger.debug('Recoloring Cell Markers')
            rgb_markers = np.zeros_like(self.frame.img)
            colors = [(np.random.randint(10, 255),
                       np.random.randint(10, 255),
                       np.random.randint(10, 255)) \
                      for _ in range(1,np.max(markers))]
            colors = [(0,0,0)] + colors + [(80,80,80)]
            for j in tqdm(range(rgb_markers.shape[0])):
                for i in range(rgb_markers.shape[1]):
                    rgb_markers[j,i] = colors[markers[j,i]]
            
            # plot all the intermediate images
            fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
            axs = axs.ravel()
            axs[0].imshow(self.frame.img)
            axs[0].set_title('Original Frame')
            axs[1].matshow(img_objs, cmap='gray')
            axs[1].set_title('Proc. Stain for Objs only')
            axs[2].imshow(img_dist, cmap='gray')
            axs[2].set_title('Distance Transform')
            axs[3].matshow(img_thresh, cmap='gray')
            axs[3].set_title('Proc. Stain for All Cells Parts')
            axs[4].imshow(frame.img)
            axs[4].set_title('Original Stain')
            axs[5].matshow(unknown, cmap='gray')
            axs[5].set_title('Unknown Pixel Data')                        
            axs[6].imshow(img_minmax, cmap='gray')
            axs[6].set_title('Min-Max Filtered Dist. Transform')
            axs[6].plot(candidates[1], candidates[0], 'x', color='red')
            axs[7].matshow(rgb_markers, cmap='rainbow')
            axs[7].set_title('Instance Segmented Cells')
            fig.tight_layout()
        #
        # end of debugging plots
        
        return cells
    #
    # end of segment_cells
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
                    'lvl': self.frame.lvl,
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
            cell.lvl = self.frame.lvl
            cell.order = 1
         
        return cells
    #
    # end of markers_to_cells

    def get_thresh(self, img, threshold, mask, close_r, open_r):
        img = img.copy()
        if not mask is None:
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
        fpath = sana_io.create_filepath(
            self.fname, ext='.png', suffix=suffix, fpath=odir)
        np.save(fpath, arr)
    #
    # end of save_array

    def save_params(self, odir, params):
        fpath = sana_io.create_filepath(self.fname, ext='.csv', fpath=odir)
        params.write_data(fpath)
    #
    # end of save_params

    def save_ao_arr(self, odir, frame, suffix=''):
        fpath = sana_io.create_filepath(
            self.fname, ext='.dat', suffix=suffix, fpath=odir)
        
        # make sure image is a boolean array
        img = frame.img
        img[img != 0] = 1
        img = img.astype(bool)
        if len(img.shape) == 3:
            img = img[:,:,0]
            
        # pack the bools into bytes (compresses to 1/8 the size)
        arr = np.packbits(img, axis=-1, bitorder='little')

        # compress and save the array
        compressed_arr = io.BytesIO()
        np.savez_compressed(compressed_arr, arr)
        with open(fpath, 'wb') as fp:
            fp.write(compressed_arr.getbuffer())
        
#
# end of Processor

def run_segment_markers(pid, ret, args):
    ret[pid] = segment_markers(**args)
    
def segment_markers(markers, lvl, converter, st, en):
    cells = []
    z = np.zeros((markers.shape[0], markers.shape[1]), np.uint8)
    o = z.copy() + 1
    marker_vals = np.sort(np.unique(markers))[1:]
    for val in marker_vals:
        x = np.where(markers == val, o, z)
        f = Frame(x, lvl, converter)
        f.get_contours()
        f.filter_contours()
        bodies = f.get_body_contours()
        if len(bodies) != 0:
            cell = bodies[0].polygon.connect()
            cell[:,1] += st
            cells.append(cell)
    return cells

#
# end of file
