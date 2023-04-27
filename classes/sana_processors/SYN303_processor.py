
# system modules
import os

# installed modules
import cv2
import numpy as np
from sana_logger import SANALogger
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
from sana_frame import Frame, create_mask, overlay_thresh
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import SYN303Classifier
from sana_loader import Loader

# debugging modules
from matplotlib import pyplot as plt

class SYN303Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(SYN303Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):
        # generate the neuronal and glial severity results

        self.logger.info('Running SYN303 Processor...')

        self.generate_masks(main_roi, sub_rois)

        # either use the cmdl input value or a pre-defined value from before
        # NOTE: this pre-defined value was picked from analyzing multiple slides
        #        in QuPath w/ varying intensities and pathology severity
        if not hasattr(self, 'manual_dab_threshold'):
            self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params, save_images=False)
        
        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90, save_images=False)

        # self.run_lb_detection(odir, detection_odir, first_run, params, 0.60)

        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save params to csv
        self.save_params(odir, params)        
    #
    # end of run

    # gets the avg value within the polygon
    def get_confidence(self, poly, frame):
        msk = create_mask([poly],frame.size(),lvl=frame.lvl,converter=frame.converter)
        return np.mean(frame.img[msk.img[:,:,0]==1])
    #
    # end of get_confidence

    def run_lb_detection(self, odir, detection_odir, first_run, params, softmax_prob_thresh=0.5):   
        self.logger.info('Running LB detections...')
        self.logger.debug('Softmax Prob. Thresh: %0.2f' %softmax_prob_thresh)

        # decode the model to get activations of each class
        model = SYN303Classifier(self.frame)
        wc_activation = model.run()
        wc_probs = softmax(wc_activation, axis=0)
        
        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_PROBS.npy'))
        np.save(ofname, wc_activation)
        
        if self.logger.plots:
             # debug WC activations for each of the classes
            class_dict = {
                0: 'Background',
                1: 'Lewy-Body'
            }
            fig, ax = plt.subplots(1,1)
            fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            ax.imshow(self.frame.img)
            fig, axs = plt.subplots(1,2)
            axs = axs.ravel()
            for i in range(len(axs)):
                axs[i].imshow(wc_probs[i,:,:])
                axs[i].set_title(class_dict[i])
            fig.suptitle('WildCat Class Activation Maps')
            plt.tight_layout()
            plt.show()
        
        relevant_lb_dab_msk, dab_thresh = self.process_dab(self.dab,
            run_normalize = True,
            scale = 1.0,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            close_r = 0,
            open_r = 13, #always an odd number (7 < open_r < 13)
            mask = self.main_mask,
            debug = self.logger.plots
        )
        
        # relevant_ln_dab_msk, dab_thresh = self.process_dab(
        #     self.dab,
        #     run_normalize = True,
        #     scale = 1.0,
        #     mx = 90, #default: 90, experiment w/ mx = 100-110
        #     close_r = 0,
        #     open_r = 0, #always an odd number (7 < open_r < 13)
        #     mask = self.main_mask,
        #     debug = self.logger.plots
        # )
        
        self.logger.info('DAB Thresh: %d' %dab_thresh)

        # get the lewy body pixel activations
        lb_activation = wc_probs[1,:,:]
        lb_activation = Frame(
            lb_activation, lvl=self.dab.lvl, converter=self.dab.converter
        )
        size = self.frame.size()
        lb_activation.img = cv2.resize(
            lb_activation.img, dsize=(int(size[0]), int(size[1])),
            interpolation=cv2.INTER_NEAREST
        )
        # ln_activation = wc_activation[2,:,:]
        # ln_activation = Frame(
        #     ln_activation, lvl=self.dab.lvl, converter=self.dab.converter
        # )
        # ln_activation.img = cv2.resize(
        #     ln_activation.img, dsize=(int(size[0]), int(size[1])),
        #     interpolation=cv2.INTER_NEAREST
        # )

        # get the pixel predictions
        # NOTE: we remove predictions that aren't very confident to remove background!
        thresh = softmax_prob_thresh
        wc_preds = np.argmax(wc_probs, axis=0)
        wc_preds[np.max(wc_probs, axis=0) < thresh] = 0
        lb_preds = Frame((wc_preds == 1).astype(np.uint8), self.dab.lvl, self.dab.converter)
        lb_preds.img = cv2.resize(lb_preds.img, dsize=tuple(self.frame.size()),
                                  interpolation=cv2.INTER_NEAREST)[:,:,None]
        # ln_preds = Frame((wc_preds == 2).astype(np.uint8), self.dab.lvl, self.dab.converter)

        # ln_preds.img = cv2.resize(ln_preds.img, dsize=tuple(self.frame.size()),
        #                           interpolation=cv2.INTER_NEAREST)[:,:,None]

        # LB mask is the pixels LB was predicted and DAB is strong
        lb_msk = lb_activation.copy()
        lb_msk.img = ((lb_preds.img == 1) & (relevant_lb_dab_msk.img == 1)).astype(np.uint8)

        # LN mask is the pixels LN was predicted and DAB is strong
        # TODO: need a separate relevant_dab_mask since the opening filter rmeoves all LNs!!!
        # ln_msk = ln_activation.copy()
        # ln_msk.img = ((ln_preds.img == 1) & (relevant_ln_dab_msk.img == 1)).astype(np.uint8)

        # get the contours/polygons from the LB mask
        lb_msk.get_contours()
        lb_msk.filter_contours(min_body_area=0/self.dab.converter.mpp) #default: 20/mpp
        lb_contours = lb_msk.get_body_contours()
        lbs = [contour.polygon for contour in lb_contours]
        
        # get the contours/polygons from the LN mask
        # ln_msk.get_contours()
        # ln_msk.filter_contours(min_body_area=0/self.dab.converter.mpp) #default: 20/mpp
        # ln_contours = ln_msk.get_body_contours()
        # lns = [contour.polygon for contour in ln_contours]

        # lbs = []
        # # apply a coarseness mask
        # for i, lb in enumerate(old_lbs):
        #     # - load local DAB image using bounding box of LB polygon
        #     loc, size = lb.bounding_box()
        #     local_dab = self.dab.copy()
        #     lb_msk_c = lb_msk.copy()
        #     local_dab.crop(loc,size)
        #     lb_msk_c.crop(loc,size)
        #     # self.save_frame(odir, local_dab, 'Local_DAB_LB_%d' %i)
        #     # print('Saving local DAB...')
        #     # print(odir,'\n')
            
        #     local_dab.img = 255*(np.log(local_dab.img[:,:,0],where=local_dab.img[:,:,0]!=0)/np.log(255))
            
        #     # Chan-Vese Segmentation Method
        #     # phi and energies used for testing/debugging chan_vese()
        #     segmentation_msk, phi, energies = chan_vese(local_dab.img,
        #         mu=0.2,
        #         lambda1=20,
        #         lambda2=40,
        #         dt=2.5,
        #         tol=1e-7,
        #         max_num_iter=8,
        #         init_level_set='checkerboard',
        #         extended_output=True
        #     )

        #     # send segmentation img to a Frame obj
        #     local_lb_msk = Frame(segmentation_msk.copy().astype(np.uint8),lvl=self.dab.lvl,converter=self.dab.converter)

        #     if self.logger.plots:
        #         # Plot outputs from Chan-Vese
        #         # fig, axs = plt.subplots(2,2)
        #         # fig.suptitle('Chan-Vese Segmentation of Processed DAB')
        #         # axs = axs.ravel()
        #         # axs[0].set_title('Orig. Local DAB')            
        #         # axs[0].imshow(local_dab.img,cmap='gray')
        #         # axs[1].set_title('Segmentation')
        #         # axs[1].imshow(segmentation_msk,cmap='gray')
        #         # axs[2].set_title('Level Set')
        #         # axs[2].imshow(phi,cmap='gray')
        #         # axs[3].set_title('Energy/iter. (to be minimized)')
        #         # axs[3].plot(energies)
        #         # fig.tight_layout()

        #         # Plot outputs and compare to old method
        #         fig, axs = plt.subplots(2,2)
        #         fig.suptitle('Chan-Vese Segmentation of Processed DAB')
        #         axs = axs.ravel()
        #         axs[0].set_title('Orig. Local DAB')            
        #         axs[0].imshow(local_dab.img,cmap='gray')
        #         axs[1].set_title('Segmentation')
        #         axs[1].imshow(segmentation_msk,cmap='gray')
        #         axs[2].set_title('Old LB Mask')
        #         axs[2].imshow(lb_msk_c.img,cmap='gray')
        #         axs[3].set_title('LB Mask')
        #         axs[3].imshow(local_lb_msk.img)
        #         fig.tight_layout()
        #         # plt.show()

        #     # - find contours on that thresholded image surrounding the LB polygon
        #     local_lb_msk.get_contours()
        #     local_lb_msk.filter_contours(min_body_area=10/self.dab.converter.mpp) #default: 20/mpp
        #     lb_contour = local_lb_msk.get_body_contours()
        #     new_lbs = [c.polygon for c in lb_contour]
        #     # new_lbs = [transform_inv_poly(x, loc, params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in new_polys]
        #     lb.translate(loc)

        #     # Plot the new LBs on the DAB and Orig img
        #     if self.logger.plots:
        #         local_frame = self.frame.copy()
        #         local_frame.crop(loc,size)
        #         fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
        #         fig.suptitle('Comparison of Orig LB Poly vs Chan-Vese LB Poly')
        #         axs[0].set_title('Before CV')
        #         axs[0].imshow(local_frame.img)
        #         plot_poly(axs[0],lb,color='red')
        #         axs[1].set_title('After CV')
        #         axs[1].imshow(local_frame.img)
        #         axs[2].set_title('After EFD')
        #         axs[2].imshow(local_frame.img)
        #         for i, new_lb in enumerate(new_lbs):
        #             plot_poly(axs[1],new_lb,color='red')
        #         plt.show()
            
        #     # translate before saving
        #     [p.translate(-loc) for p in new_lbs]

        #     # - replace the LB polygon w/ this contour 
        #     if new_lbs:
        #         lbs.append(new_lbs[-1])
        #
        # end of Chan-Vese loop

            # * done before saving the LB annotations in run_lb_detections

        # convert polygons to annotations and calculate confidence
        bid, antibody, region, roi_name = odir.split('/')[-4:]
        tile = roi_name.split('_')[-1]
        
        lb_annos = []
        for lb in lbs:

            # confidence is the average LB activation inside the polygon
            conf = self.get_confidence(lb, lb_activation)
            
            fname = bid+'_'+antibody+'_'+region+'_'+tile
            lb_anno = lb.to_annotation(
                fname, class_name='LB detection', confidence=conf
            )
            lb_annos.append(lb_anno)

        # ln_annos = []
        # for ln in lns:

        #     # confidence is the average LB activation inside the polygon
        #     conf = self.get_confidence(ln, ln_activation)

        #     fname = bid+'_'+antibody+'_'+region+'_'+tile
        #     ln_anno = ln.to_annotation(
        #         fname, class_name='LN detection', confidence=conf
        #     )
        #     ln_annos.append(ln_anno)

        lb_poly_mask = create_mask(lb_annos, self.frame.size(), self.frame.lvl, self.frame.converter)
        # ln_poly_mask = create_mask(ln_annos, self.frame.size(), self.frame.lvl, self.frame.converter)

        # run the %AO of the detections
        results = self.run_ao(lb_preds)
        params.data['lb_wc_ao'] = results['ao']
        self.save_signals(odir, results['signals']['main_deform'], 'LB')
        
        results = self.run_ao(lb_poly_mask)
        params.data['lb_poly_ao'] = results['ao']
        self.save_signals(odir, results['signals']['main_deform'], 'LB_POLY')        
        
        # results = self.run_ao(ln_preds)
        # params.data['ln_wc_ao'] = results['ao']
        # self.save_signals(odir, results['signals']['main_deform'], 'LN')        
        
        # results = self.run_ao(ln_poly_mask)
        # params.data['ln_poly_ao'] = results['ao']
        # self.save_signals(odir, results['signals']['main_deform'], 'LN_POLY')

        if len(lb_annos)>0 and self.logger.plots:
            fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
            fig.suptitle('Debugging Plot of an LB Detection')
            axs = axs.ravel()

            # plot the original image
            axs[0].imshow(self.frame.img) 
            axs[0].set_title('Orig. Img')

            # plot the strong DAB in the image
            axs[1].imshow(relevant_lb_dab_msk.img)
            axs[1].set_title('DAB mask')
            
            alpha = 0.4
            overlay = overlay_thresh(self.frame, lb_preds, alpha=alpha, color='red')
            # overlay = overlay_thresh(overlay, ln_preds, alpha=alpha, color='blue')
            axs[2].imshow(overlay.img)
            axs[2].set_title('WC Pixel Predictions')
            [plot_poly(axs[2], lb, color='black') for lb in lbs]
            # [plot_poly(axs[2], lb, color='yellow') for lb in lns]

            fig, axs = plt.subplots(1,2)
            axs[0].imshow(wc_probs[0])
            axs[1].imshow(wc_probs[1])
            
            plt.show()
        #
        # end of debugging
        
        # transform detections to the original slide coordinate system
        annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in lb_annos]
        
        # set pixel lvl to 0
        [self.dab.converter.rescale(x, 0) for x in annos]        

        # finally, write the annoations to a file
        afile = os.path.basename(self.fname).replace('.svs','.json')
        anno_fname = detection_odir+'/'+afile
        if not os.path.exists(anno_fname) or first_run:
            sana_io.write_annotations(anno_fname, annos)
            first_run = False
        else:
            sana_io.append_annotations(anno_fname, annos)
        
    # 
    # end of run_lb_detection#
# end of R13Processor

#
# end of file
