
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, overlay_thresh
from sana_heatmap import Heatmap
from sana_geo import Point

# debugging modules
from matplotlib import pyplot as plt

TSIZE = Point(400, 100, is_micron=False, lvl=0)
TSTEP = Point(50, 50, is_micron=False, lvl=0)

# generic processor class, sets the main attributes and holds
# functions for generating data from processed Frames
class Processor:
    def __init__(self, fname, frame, roi_type="", debug=False):
        self.fname = fname
        self.frame = frame
        self.roi_type = roi_type
        self.debug = debug
    #
    # end of constructor

    def mask_frame(self, main_roi, sub_rois=[]):
        
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

        # apply the main mask to the frame
        # TODO: we need to check if this is needed
        if self.frame.slide_color is None:
            self.frame.mask(self.main_mask)
        else:
            self.frame.mask(self.main_mask, self.frame.slide_color)
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
        if self.roi_type == 'GM':
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
        sub_deformed_feats = heatmap.deform(feats, self.sub_masks))

        # calculate the signals as the average over the columns
        signals = {
            'normal': np.mean(feats, axis=2)[None,...],
            'main_deform': np.mean(main_deformed_feats, axis=3),
            'sub_deform': np.mean(sub_deformed_feats, axis=3),
        }

        fig, axs = plt.subplots(1,4)
        axs[0].imshow(frame.img)
        axs[1].imshow(feats[0])
        axs[2].imshow(main_deformed_feats[0])
        axs[3].imshow(sub_deformed_feats[0])

        fig, axs = plt.subplots(1,3)
        axs[0].plot(signals['normal'][0][0])
        axs[1].plot(signals['main_deform'][0][0])
        for i in range(signals['sub_deform'].shape[0]):
            n = np.arange(signals['sub_deform'].shape[2])
            x = np.arange(n) + i*n
            axs[2].plot(x, signals['sub_deform'][i][0])
        
        plt.show()
        
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
