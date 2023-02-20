
# system modules
import os

# installed modules
import cv2
import numpy as np
from sana_logger import SANALogger
from sana_frame import overlay_thresh
from tqdm import tqdm
from PIL import Image
import json
from scipy.special import softmax
from scipy.ndimage.filters import generic_filter
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import pyefd 

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, plot_poly, transform_inv_poly, Polygon
from sana_frame import Frame, create_mask, overlay_thresh, frame_like
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import R13Classifier
from sana_loader import Loader
from sana_params import Params

# debugging modules
from matplotlib import pyplot as plt

class R13Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(R13Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):
        self.logger.info('Running R13 Processor...')
        
        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        if self.run_wildcat:
            self.run_lb_detection(odir, roi_odir, first_run, params, 0.85)

        # generate the auto AO results
        # TODO: define a minimum threshold value!
        self.run_auto_ao(odir, params, scale=1.0, mx=90)

        # either use the cmdl input value or a pre-defined value from before
        # NOTE: this pre-defined value was picked from analyzing multiple slides
        #        in QuPath w/ varying intensities and pathology severity
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    # gets the avg value within the polygon
    def get_confidence(self, poly, frame):
        msk = create_mask([poly],frame.size(),lvl=frame.lvl,converter=frame.converter)
        return np.mean(frame.img[msk.img[:,:,0]==1])
    #
    # end of get_confidence

    def run_lb_detection(self, odir, roi_odir, first_run, params, softmax_prob_thresh=0.5):   
        self.logger.info('Running LB detections...')
        self.logger.debug('Softmax Prob. Thresh: %0.2f' %softmax_prob_thresh)

        # decode the model to get activations of each class
        model = R13Classifier(self.frame)
        wc_activation = model.run()
        wc_probs = softmax(wc_activation, axis=0)
        
        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_R13.npy'))
        np.save(ofname, wc_activation)
        
        if self.logger.plots:
             # debug WC activations for each of the classes
            class_dict = {
                0: 'Artifact',
                1: 'Background',
                2: 'Lewy-Body',
                3: 'Lewy-Neurite'
            }
            fig, ax = plt.subplots(1,1)
            fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            ax.imshow(self.frame.img)
            fig, axs = plt.subplots(2,2)
            axs = axs.ravel()
            for i in range(len(axs)):
                axs[i].imshow(wc_activation[i,:,:])
                axs[i].set_title(class_dict[i])
            fig.suptitle('WildCat Class Activation Maps')
            plt.tight_layout()
            plt.show()
        
        relevant_dab_msk, dab_thresh = self.process_dab(self.dab,
            run_normalize = True,
            scale = 1.0,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            close_r = 0,
            open_r = 13, #always an odd number (7 < open_r < 13)
            mask = self.main_mask,
            debug = self.logger.plots
        )
        relevant_ln_dab_msk, dab_thresh = self.process_dab(
            self.dab,
            run_normalize = True,
            scale = 1.0,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            close_r = 0,
            open_r = 0, #always an odd number (7 < open_r < 13)
            mask = self.main_mask,
            debug = self.logger.plots
        )
        
        self.logger.info('DAB Thresh: %d' %dab_thresh)

        # 1 for dice loss and 2 for SoftMargin (selecting LB activation map)
        lb_activation = wc_activation[2,:,:]
        lb_softmax = wc_probs[2,:,:]

        tup_size = self.frame.size()
        tup_size = (tup_size[0], tup_size[1])
        lb_threshold = 0.4        
        lb_mask = (lb_softmax > lb_threshold).astype(np.uint8)
        lb_mask = cv2.resize(lb_mask, tup_size, interpolation=cv2.INTER_NEAREST)
        lb_mask = frame_like(self.frame, lb_mask)
        results = self.run_ao(lb_mask)
        params.data['lb_wc_ao'] = results['ao']
        params.data['lb_wc_sub_aos'] = results['sub_aos']
        
        # # Calculate hyp annos
        # # store DAB 
        # lb_activation = Frame(lb_activation,lvl=self.dab.lvl,converter=self.dab.converter)
        # tup_size = relevant_dab_msk.size()
        # tup_size = (tup_size[0], tup_size[1])
        # lb_activation.img = cv2.resize(lb_activation.img, tup_size, interpolation=cv2.INTER_NEAREST)
        
        # lb_softmax = Frame(lb_softmax,lvl=self.dab.lvl,converter=self.dab.converter)
        # lb_softmax.img = cv2.resize(lb_softmax.img, tup_size, interpolation=cv2.INTER_NEAREST)     

        # # threshold probs img at LB prob and store in a Frame
        # lb_msk = np.where(lb_softmax.img >= softmax_prob_thresh, relevant_dab_msk.img[:,:,0], np.zeros_like(relevant_dab_msk.img[:,:,0]))
        # lb_msk = Frame(lb_msk, lvl=self.dab.lvl, converter=self.dab.converter)

        # lb_msk.get_contours()
        # lb_msk.filter_contours(min_body_area=0/self.dab.converter.mpp) #default: 20/mpp
        # lb_contours = lb_msk.get_body_contours()
        # lbs = [contour.polygon for contour in lb_contours]        
        # #old_lbs = [contour.polygon for contour in lb_contours]
        # # lbs = []
        # # # apply a coarseness mask
        # # for i, lb in enumerate(old_lbs):
        # #     # - load local DAB image using bounding box of LB polygon
        # #     loc, size = lb.bounding_box()
        # #     local_dab = self.dab.copy()
        # #     lb_msk_c = lb_msk.copy()
        # #     local_dab.crop(loc,size)
        # #     lb_msk_c.crop(loc,size)
        # #     # self.save_frame(odir, local_dab, 'Local_DAB_LB_%d' %i)
        # #     # print('Saving local DAB...')
        # #     # print(odir,'\n')
            
        # #     local_dab.img = 255*(np.log(local_dab.img[:,:,0],where=local_dab.img[:,:,0]!=0)/np.log(255))
            
        # #     # Chan-Vese Segmentation Method
        # #     # phi and energies used for testing/debugging chan_vese()
        # #     segmentation_msk, phi, energies = chan_vese(local_dab.img,
        # #         mu=0.2,
        # #         lambda1=20,
        # #         lambda2=40,
        # #         dt=2.5,
        # #         tol=1e-7,
        # #         max_num_iter=8,
        # #         init_level_set='checkerboard',
        # #         extended_output=True
        # #     )

        # #     # send segmentation img to a Frame obj
        # #     local_lb_msk = Frame(segmentation_msk.copy().astype(np.uint8),lvl=self.dab.lvl,converter=self.dab.converter)

        # #     if self.logger.plots:
        # #         # Plot outputs from Chan-Vese
        # #         # fig, axs = plt.subplots(2,2)
        # #         # fig.suptitle('Chan-Vese Segmentation of Processed DAB')
        # #         # axs = axs.ravel()
        # #         # axs[0].set_title('Orig. Local DAB')            
        # #         # axs[0].imshow(local_dab.img,cmap='gray')
        # #         # axs[1].set_title('Segmentation')
        # #         # axs[1].imshow(segmentation_msk,cmap='gray')
        # #         # axs[2].set_title('Level Set')
        # #         # axs[2].imshow(phi,cmap='gray')
        # #         # axs[3].set_title('Energy/iter. (to be minimized)')
        # #         # axs[3].plot(energies)
        # #         # fig.tight_layout()

        # #         # Plot outputs and compare to old method
        # #         fig, axs = plt.subplots(2,2)
        # #         fig.suptitle('Chan-Vese Segmentation of Processed DAB')
        # #         axs = axs.ravel()
        # #         axs[0].set_title('Orig. Local DAB')            
        # #         axs[0].imshow(local_dab.img,cmap='gray')
        # #         axs[1].set_title('Segmentation')
        # #         axs[1].imshow(segmentation_msk,cmap='gray')
        # #         axs[2].set_title('Old LB Mask')
        # #         axs[2].imshow(lb_msk_c.img,cmap='gray')
        # #         axs[3].set_title('LB Mask')
        # #         axs[3].imshow(local_lb_msk.img)
        # #         fig.tight_layout()
        # #         # plt.show()

        # #     # - find contours on that thresholded image surrounding the LB polygon
        # #     local_lb_msk.get_contours()
        # #     local_lb_msk.filter_contours(min_body_area=10/self.dab.converter.mpp) #default: 20/mpp
        # #     lb_contour = local_lb_msk.get_body_contours()
        # #     new_lbs = [c.polygon for c in lb_contour]
        # #     # new_lbs = [transform_inv_poly(x, loc, params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in new_polys]
        # #     lb.translate(loc)

        # #     # Plot the new LBs on the DAB and Orig img
        # #     if self.logger.plots:
        # #         local_frame = self.frame.copy()
        # #         local_frame.crop(loc,size)
        # #         fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
        # #         fig.suptitle('Comparison of Orig LB Poly vs Chan-Vese LB Poly')
        # #         axs[0].set_title('Before CV')
        # #         axs[0].imshow(local_frame.img)
        # #         plot_poly(axs[0],lb,color='red')
        # #         axs[1].set_title('After CV')
        # #         axs[1].imshow(local_frame.img)
        # #         axs[2].set_title('After EFD')
        # #         axs[2].imshow(local_frame.img)
        # #         for i, new_lb in enumerate(new_lbs):
        # #             plot_poly(axs[1],new_lb,color='red')
        # #         plt.show()
            
        # #     # translate before saving
        # #     [p.translate(-loc) for p in new_lbs]

        # #     # - replace the LB polygon w/ this contour 
        # #     if new_lbs:
        # #         lbs.append(new_lbs[-1])

        # #     # * done before saving the LB annotations in run_lb_detections

        # # convert polygons to annotations and calculate confidence
        # bid, antibody, region, roi_name = odir.split('/')[-4:]
        # tile = roi_name.split('_')[-1]
        # lb_annos = []
        # for lb in lbs:

        #     # confidence is the average LB activation inside the polygon
        #     conf = self.get_confidence(lb, lb_activation)
            
        #     fname = bid+'_'+antibody+'_'+region+'_'+tile
        #     lb_anno = lb.to_annotation(
        #         fname, class_name='LB detection', confidence=conf
        #     )
        #     lb_annos.append(lb_anno)

        # # transform detections to the original slide coordinate system
        # lb_annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in lb_annos]

        # # TODO: this is broken
        # # if len(lb_annos)>0 and self.logger.plots:
        # #     fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
        # #     fig.suptitle('Debugging Plot of an LB Detection')
        # #     axs = axs.ravel()

        # #     # plot the original image
        # #     axs[0].imshow(self.frame.img) 
        # #     axs[0].set_title('Orig. Img')

        # #     # plot the strong DAB in the image
        # #     axs[1].imshow(relevant_dab_msk.img)
        # #     axs[1].set_title('DAB mask')

        # #     alpha = 0.4
        # #     lb_preds = frame_like(self.frame, wc_probs[2] > softmax_prob_thresh)
        # #     ln_preds = frame_like(self.frame, wc_probs[3] > softmax_prob_thresh)
        # #     overlay = overlay_thresh(self.frame, lb_preds, alpha=alpha, color='red')
        # #     overlay = overlay_thresh(overlay, ln_preds, alpha=alpha, color='blue')
        # #     axs[2].imshow(overlay.img)
        # #     axs[2].set_title('WC Pixel Predictions')
        # #     [plot_poly(axs[2], lb, color='black') for lb in lbs]
        # #     [plot_poly(axs[2], lb, color='yellow') for lb in lns]            

        # #     fig, axs = plt.subplots(1,3)
        # #     axs[0].imshow(wc_probs[0])
        # #     axs[1].imshow(wc_probs[1])
        # #     axs[2].imshow(wc_probs[2])
            
        # #     plt.show()
        # # #
        # # # end of debugging

        # # set pixel lvl to 0
        # [self.dab.converter.rescale(x, 0) for x in lb_annos]        

        # # finally, write the annoations to a file
        # afile = os.path.basename(self.fname).replace('.svs','.json')
        # anno_fname = roi_odir+'/'+afile
        # if not os.path.exists(anno_fname) or first_run:
        #     sana_io.write_annotations(anno_fname, lb_annos)
        #     first_run = False
        # else:
        #     sana_io.append_annotations(anno_fname, lb_annos)
        
    # 
    # end of run_lb_detection
#
# end of R13Processor

#
# end of file
