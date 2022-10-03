# TODO: 


# system modules
import os

# installed modules
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from scipy.special import softmax

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, plot_poly, transform_inv_poly
from sana_frame import Frame, create_mask, overlay_thresh
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import R13Classifier
from sana_loader import Loader

# debugging modules
from matplotlib import pyplot as plt

class R13Classifier(HDABProcessor):
    def __init__(self, fname, frame, debug=False):
        super(R13Classifier, self).__init__(fname, frame, debug)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):
        # generate the neuronal and glial severity results

        self.generate_masks(main_roi, sub_rois)

        self.run_lb_detection(odir, roi_odir, first_run, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90)
    #
    # end of run

    # gets the avg value within the polygon
    def get_confidence(self, poly, frame):
        msk = create_mask([poly],frame.size(),lvl=frame.lvl,converter=frame.converter)
        return np.mean(frame.img[msk.img[:,:,0]==1])
    #
    # end of get_confidence

    def run_lb_detection(self, odir, roi_odir, first_run, debug, params):
        # debug = False

        if debug:
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = np.ravel(axs)

        model = R13Classifier(self.frame)
        wc_activation = model.run()
        
        wc_softmax = softmax(wc_activation,axis=0)

        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_R13.npy'))
        np.save(ofname, wc_activation)
        
        # # debug WC activations for each of the classes
        # class_dict = {
        #     0: 'Artifact',
        #     1: 'Background',
        #     2: 'Lewy-Body',
        #     3: 'Lewy-Neurite'
        # }
        # fig, axs = plt.subplots(2,2)
        # axs = axs.ravel()
        # for i in range(wc_softmax.shape[0]):
        #     axs[i].imshow(wc_softmax[i,:,:],vmin=0,vmax=1)
        #     axs[i].set_title(class_dict[i])
        # plt.show()
        # exit()

        relevant_dab_msk, dab_thresh = self.process_dab(self.dab,
            run_normalize = True,
            scale = 1.0,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            close_r = 0,
            open_r = 13, #always an odd number (7 < open_r < 13)
            mask = self.main_mask,
            debug = False
            )
        # print('DAB Thresh:',dab_thresh)

        lb_activation = wc_activation[2,:,:]
        lb_softmax = wc_softmax[2,:,:]

        # Calculate hyp annos
        # store DAB 
        lb_activation = Frame(lb_activation,lvl=self.dab.lvl,converter=self.dab.converter)
        lb_activation.img = cv2.resize(lb_activation.img, relevant_dab_msk.size(),interpolation=cv2.INTER_NEAREST)
        
        lb_softmax = Frame(lb_softmax,lvl=self.dab.lvl,converter=self.dab.converter)
        lb_softmax.img = cv2.resize(lb_softmax.img, relevant_dab_msk.size(),interpolation=cv2.INTER_NEAREST)

        # fig, axs = plt.subplots(1,3)
        # axs[0].imshow(self.dab.img)
        # axs[0].set_title('DAB Img')
        # axs[1].imshow(wc.img)
        # axs[1].set_title('Nonsoftmax')
        # axs[2].imshow(wc_soft.img)
        # axs[2].set_title('Softmax')
        # plt.show()
        # exit()        

        # threshold probs img at LB prob and store in a Frame
        # img = np.where(wc.img==1, dab_img.img[:,:,0], np.zeros_like(dab_img.img[:,:,0]))
        lb_msk = np.where(lb_softmax.img >= 0.8, relevant_dab_msk.img[:,:,0], np.zeros_like(relevant_dab_msk.img[:,:,0]))
        # img = np.where(wc.img < dab_thresh, 0, 1).astype(np.uint8)
        # img = np.where(wc.img < -15, 0, dab_img.img[:,:,0]).astype(np.uint8)
        
        lb_msk = Frame(lb_msk, lvl=self.dab.lvl, converter=self.dab.converter)

        lb_msk.get_contours()
        lb_msk.filter_contours(min_body_area=0/self.dab.converter.mpp) #default: 20/mpp
        lb_contours = lb_msk.get_body_contours()
        lbs = [contour.polygon for contour in lb_contours]

        # # write out hyp annos with lb detections
        # fig, axs = plt.subplots(2,2)
        # axs = np.ravel(axs)
        # axs[0].imshow(self.dab.img)
        # axs[0].set_title('Orig DAB Img')
        # axs[1].imshow(wc.img)
        # axs[1].set_title('WC Img')
        # axs[2].imshow(dab_img.img)
        # axs[2].set_title('Processed DAB Img')
        # axs[3].imshow(lb_msk.img)
        # axs[3].set_title('LB Msk Img')
        # plt.show()
        # exit()

        # convert polygons to annotations and calculate confidence
        bid, antibody, region, roi_name = odir.split('/')[-4:]
        tile = roi_name.split('_')[-1]

        lb_annos = []
        for lb in lbs:
            fname = bid+'_'+antibody+'_'+region+'_'+tile
            conf = self.get_confidence(lb, lb_activation)
            lb_annos.append(lb.to_annotation(fname,class_name='LB detection',confidence=conf))


        if debug and len(lb_annos)>0:
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.ravel()
            for lb in lbs:
                plot_poly(axs[0],lb,color='red')
                plot_poly(axs[1],lb,color='red')
                plot_poly(axs[2],lb,color='red')
                plot_poly(axs[3],lb,color='red')
     
            axs[0].imshow(self.frame.img) 
            axs[0].set_title('Orig. Img')

            axs[1].imshow(lb_softmax.img,vmin=0,vmax=1)
            axs[1].set_title('WildCat Frame')

            axs[2].imshow(relevant_dab_msk.img)
            axs[2].set_title('Proc. DAB Mask')

            axs[3].imshow(lb_msk.img)
            axs[3].set_title('LB Mask')
            plt.show()

        # transform detections to the original slide coordinate system
        lb_annos = [transform_inv_poly(x, params.data['loc'], params.data['padding'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in lb_annos]

        # set pixel lvl to 0
        [self.dab.converter.rescale(lb, 0) for lb in lb_annos]        

        afile = os.path.basename(self.fname).replace('.svs','.json')
        anno_fname = roi_odir+'/'+afile
        if not os.path.exists(anno_fname) or first_run:
            sana_io.write_annotations(anno_fname, lb_annos)
            first_run = False
        else:
            sana_io.append_annotations(anno_fname, lb_annos)
        
    # 
    # end of run_lb_detection
#
# end of R13Processor

#
# end of file
