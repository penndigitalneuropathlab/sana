
# system modules
import os

# installed modules
import numpy as np
import cv2

# custom modules
import sana_io
from sana_frame import Frame
from sana_color_deconvolution import StainSeparator
from sana_antibody_processor import Processor
from sana_thresholds import max_dev
from sana_geo import transform_inv_poly

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

class HEProcessor(Processor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(HEProcessor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):
        self.generate_masks(main_roi, sub_rois)

        #fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
        #axs = axs.ravel()
        
        #axs[0].imshow(self.frame.img)
        #axs[3].imshow(self.frame.img)
        
        self.frame.to_gray()
        self.frame.img = 255 - self.frame.img
        #axs[1].imshow(self.frame.img)

        blur = cv2.GaussianBlur(self.frame.img[:,:,0], (71,71), 0)
        self.frame.img[:,:,0] -= blur
        self.frame.img[self.frame.img < 0] = 0
        self.frame.img = np.rint(self.frame.img).astype(np.uint8)
        
        #axs[2].imshow(self.frame.img)
        
        hist = self.frame.histogram()
        threshold = max_dev(hist, scale=1.0, mx=255)
        cells = self.segment_cells(self.frame, threshold, disk_r=7, sigma=3, close_r=9, open_r=9, clean_r=9, n_iterations=1)
        # for cell in cells:
        #     plot_poly(axs[3], cell, color='green')
        
        anno_fname = os.path.join(odir, os.path.basename(self.fname).replace('.tif', '.json'))        
        annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']).to_annotation(anno_fname, class_name='CELL') for x in cells]

        sana_io.write_annotations(anno_fname, annos)            
        
        #plt.show()
    #
    # end of run
#
# end of HEProcessor

#
# end of file
