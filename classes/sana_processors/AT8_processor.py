
# system modules
import os

# installed modules
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, transform_inv_poly
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import MulticlassClassifier, TangleClassifier

# debugging modules
from matplotlib import pyplot as plt

class AT8Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(AT8Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        # either use the cmdl input value or a pre-defined value from before
        # NOTE: this pre-defined value was picked from analyzing multiple slides
        #        in QuPath w/ varying intensities and pathology severity
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params, save_images=False)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90, open_r=0, close_r=0, save_images=False)
        
        if self.run_wildcat:
            
            # generate the neuronal and glial severity results
            self.run_multiclass(odir, params)

            # # generate the tangle severity results
            # self.run_tangle(odir, params)

        # TODO: this should run for all HDAB processors
        self.frame.to_gray()
        self.frame.img = 255 - self.frame.img

        blur = cv2.GaussianBlur(self.frame.img[:,:,0], (71,71), 0)
        self.frame.img[:,:,0] -= blur
        self.frame.img[self.frame.img < 0] = 0
        self.frame.img = np.rint(self.frame.img).astype(np.uint8)
        
        hist = self.frame.histogram()
        threshold = max_dev(hist, scale=1.0, mx=255)
        cells = self.segment_cells(self.frame, threshold, disk_r=7, sigma=3, close_r=9, open_r=9, clean_r=9, n_iterations=1)

        anno_fname = os.path.join(odir, os.path.basename(self.fname).replace('.tif', '.json'))        
        annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']).to_annotation(anno_fname, class_name='CELL') for x in cells]
        sana_io.write_annotations(anno_fname, annos)            
            
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    # TODO: don't want to create the model everytime, should be it's own class maybe?
    def run_tangle(self, odir, params):

        model = TangleClassifier(self.frame)
        probs = model.run()

        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_TANGLE.npy'))
        np.save(ofname, probs)
    #
    # end of run_tangle

    def run_multiclass(self, odir, params):

        model = MulticlassClassifier(self.frame)
        probs = model.run()

        sm_probs = softmax(probs, axis=0)
        neuronal_preds = sm_probs[1] > 0.5
        glial_preds = sm_probs[0] > 0.5
        neuronal_ofname = os.path.join(odir, os.path.basename(self.fname).replace('.tif', '_NEURONAL.png'))
        glial_ofname = os.path.join(odir, os.path.basename(self.fname).replace('.tif', '_GLIAL.png'))
        
        Image.fromarray(neuronal_preds).save(neuronal_ofname)
        Image.fromarray(glial_preds).save(glial_ofname)        
        
        # save the output probabilities
        #ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_WILDCAT.npy'))
        #np.save(ofname, probs)
    #
    # end of run_wildcat

#
# end of AT8Processor

#
# end of file
