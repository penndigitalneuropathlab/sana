
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

        if self.run_cells:
            self.run_cell_detection(odir, params)
            
        # either use the cmdl input value or a pre-defined value from before
        # NOTE: this pre-defined value was picked from analyzing multiple slides
        #        in QuPath w/ varying intensities and pathology severity
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        #self.run_auto_ao(odir, params, scale=1.0, mx=90, open_r=0, close_r=0)
        self.run_auto_ao(odir, params, scale=1.0, mx=255, open_r=0, close_r=0, min_background=7)
        
        if self.run_wildcat:
            
            # generate the neuronal and glial severity results
            self.run_multiclass(odir, params)

            # # generate the tangle severity results
            # self.run_tangle(odir, params)
            
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    def run_cell_detection(self, odir, params):

        counter = self.hem.copy()
        #counter = self.counter.copy()
        counter.rescale(np.min(counter.img), np.max(counter.img))
        counter.anisodiff()
        
        disk_r = 5
        sigma = 5
        n_iterations = 1
        close_r = 5
        open_r = 7 
        clean_r = 9

        hist = counter.histogram()
        
        scale = 1.0
        mx = 255
        threshold = max_dev(hist, scale=scale, mx=mx, debug=self.logger.plots)
        
        cells = self.segment_cells(
            counter, threshold, disk_r, sigma, n_iterations,
            close_r, open_r, clean_r)

        density = len(cells)
        area = np.mean([x.area() for x in cells])
        params.data['cell_density'] = density
        params.data['cell_area'] = area
    
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
