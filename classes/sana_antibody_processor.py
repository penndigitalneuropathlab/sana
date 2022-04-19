
# installed modules
import numpy as np

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, create_mask, overlay_thresh

# debugging modules
from matplotlib import pyplot as plt

# generic processor class, sets the main attributes and holds
# functions for generating data from processed Frames
class Processor:
    def __init__(self, fname, frame):
        self.fname = fname
        self.frame = frame
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
            mask = create_mask(
                [sub_rois[i]],
                self.frame.size(), self.frame.lvl, self.frame.converter,
                x=0, y=255, holes=[]
            )
            self.sub_masks.append(mask)
        #
        # end of sub_masks loop

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
    def run_ao(self, frame):

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
            tmp_frame = frame.copy()
            tmp_frame.mask(sub_mask)
            sub_area = np.sum(sub_mask.img / 255)
            sub_pos = np.sum(tmp_frame.img / 255)
            sub_ao = sub_pos / sub_area
            sub_aos.append(sub_ao)
            sub_areas.append(sub_area)

        # finally, return the results
        ret = {
            'ao': ao, 'area': area,
            'sub_aos': sub_aos, 'sub_areas': sub_areas,
        }
        return ret
    #
    # end of run_ao

    def save_frame(self, odir, frame, suffix):
        fpath = sana_io.create_filepath(
            self.fname, ext='.png', suffix=suffix, fpath=odir)
        frame.save(fpath)
    #
    # end of save_frame

    def save_params(self, odir, params):
        fpath = sana_io.create_filepath(self.fname, ext='.csv', fpath=odir)
        params.write_data(fpath)
    #
    # end of save_params
#
# end of Processor

#
# end of file
