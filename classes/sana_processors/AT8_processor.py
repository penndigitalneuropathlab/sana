
# system modules
import os

# installed modules
import cv2
import numpy as np
from PIL import Image
from scipy.special import softmax
from scipy.special import expit as sigmoid

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, transform_inv_poly, plot_poly
from sana_frame import Frame, frame_like, create_mask, overlay_thresh
from sana_loader import Loader
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import MulticlassClassifier, TangleClassifier, HIPTangleClassifier, CorticalTangleClassifier

# debugging modules
from matplotlib import pyplot as plt

class AT8Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(AT8Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):
        self.logger.info('Running AT8 Processor...')


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
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90, open_r=0, close_r=0)
        
        if self.run_wildcat:
            
            # generate the neuronal and glial severity results
            # self.run_multiclass(odir, params)

            # generate the tangle severity results
            # self.run_tangle(odir, params)

            # generate tangle detector
            self.run_tangle_detection(odir, roi_odir, first_run, params, 0.3)


        # # TODO: this should run for all HDAB processors
        # self.frame.to_gray()
        # self.frame.img = 255 - self.frame.img

        # blur = cv2.GaussianBlur(self.frame.img[:,:,0], (71,71), 0)
        # self.frame.img[:,:,0] -= blur
        # self.frame.img[self.frame.img < 0] = 0
        # self.frame.img = np.rint(self.frame.img).astype(np.uint8)
        
        # hist = self.frame.histogram()
        # threshold = max_dev(hist, scale=1.0, mx=255)
        # cells = self.segment_cells(self.frame, threshold, disk_r=7, sigma=3, close_r=9, open_r=9, clean_r=9, n_iterations=1)

        # anno_fname = os.path.join(odir, os.path.basename(self.fname).replace('.tif', '.json'))        
        # annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']).to_annotation(anno_fname, class_name='CELL') for x in cells]
        # sana_io.write_annotations(anno_fname, annos)            
            
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    # gets the avg value within the polygon
    def get_confidence(self, poly, frame):
        msk = create_mask([poly],frame.size(),lvl=frame.lvl,converter=frame.converter)
        return np.mean(frame.img[msk.img[:,:,0]==1]), np.std(frame.img[msk.img[:,:,0]==1])
    #
    # end of get_confidence

    
    def run_tangle_detection(self, odir, roi_odir, first_run, params, softmax_prob_thresh=0.5):   
        self.logger.info('Running Tangle detections...')
        self.logger.debug('Softmax Prob. Thresh: %0.2f' %softmax_prob_thresh)
        
        roi_class = odir.split('/')[-1].split('_')[0]
        if roi_class in ['GM']:
            self.logger.debug('Deploying Cortical AT8 Tangle Classifier...')
            model = CorticalTangleClassifier(self.frame)
        elif roi_class in ['CA1', 'CA2', 'CA3', 'CA4', 'Subiculum', 'DG']:
            self.logger.debug('Deploying HIP AT8 Tangle Classifier...')
            model = HIPTangleClassifier(self.frame)
        else:
            self.logger.info('No AT8 classifier found...')
            return

            

        # decode the model to get activations of each class
        wc_activation = model.run()
        # self.logger.debug('Shape of model output: %s',str(wc_activation.shape))
        # print(self.frame.img.shape)
        wc_probs = softmax(wc_activation, axis=0)
    
        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_PROBS.npy'))
        np.save(ofname, wc_activation)

        # [2,:,:] to select Tangle predictions from model output
        tangle_activation = wc_activation[2,:,:]
        tangle_softmax = wc_probs[2,:,:]

        # tangle_activation = cv2.resize(tangle_activation, self.dab.img[:,:,0].shape,interpolation=cv2.INTER_NEAREST)            
        # tangle_activation = 255*softmax(tangle_activation)
        # tangle_activation = 255*((tangle_activation - np.min(tangle_activation))/(np.max(tangle_activation) - np.min(tangle_activation)))
        # normed_dab = ((self.dab.img[:,:,0] - np.min(self.dab.img[:,:,0])) * (1/(np.max(self.dab.img[:,:,0]) - np.min(self.dab.img[:,:,0])) * 255))
        # dab_scaled_activation = Frame((normed_dab*tangle_activation),lvl=self.dab.lvl,converter=self.dab.converter)
        # # print(np.min(dab_scaled_activation.img),np.max(dab_scaled_activation.img))
        # dab_scaled_activation.img = ((dab_scaled_activation.img - np.min(dab_scaled_activation.img)) * (1/(np.max(dab_scaled_activation.img) - np.min(dab_scaled_activation.img)) * 255))

        if self.logger.plots:
            # frame resizing 
            # orig_img = cv2.resize(self.frame.img, tangle_activation.shape, interpolation=cv2.INTER_NEAREST)
            # print(self.frame.img.shape)
            # print(wc_activation[2,:,:].shape)
            # make this into a function --> visual_wc_probs(class_dict)
            # debug WC activations for each of the classes
            class_dict = {
                0: 'Background',
                1: 'GM NP',
                2: 'GM Tangle',
                3: 'GM Thread'
            }
            # fig, ax = plt.subplots(1,1)
            # ax.imshow(orig_img)
            # ax.set_title('Orig. Frame')
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            axs = axs.ravel()
            axs[0].imshow(self.frame.img)
            axs[0].set_title('Orig. Frame')
            for i in range(len(axs)-1):
                # axs[i+1].imshow(wc_probs[i,:,:])
                resized_activation = cv2.resize(wc_probs[i,:,:],self.frame.img[:,:,0].shape,interpolation=cv2.INTER_NEAREST)
                axs[i+1].imshow(resized_activation)
                axs[i+1].set_title(class_dict[i])
            fig.suptitle('WildCat Class Activation Maps')
            plt.tight_layout()
            
            # fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            # fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            # axs = axs.ravel()
            # axs[0].imshow(orig_img)
            # axs[0].set_title('Orig. Frame')
            # for i in range(len(axs)):
            #     axs[i].imshow(sigmoid(wc_activation[i,:,:]))
            #     axs[i].set_title(class_dict[i])
            # fig.suptitle('WildCat Class Activation Maps')
            # plt.tight_layout()

            # create new processing image (scale DAB intensity by tangle activation)

            # print(tangle_activation.shape)
            # print(self.dab.img[:,:,0].shape)
            
            # print('tangle activation min/max:',np.min(tangle_activation),np.max(tangle_activation))
            # print('DAB img min/max:',np.min(self.dab.img),np.max(self.dab.img))
            # print('tangle*DAB img min/max:',np.min(dab_scaled_activation.img),np.max(dab_scaled_activation.img))

            # plot new processing image
            # fig, axs = plt.subplots(2,2)
            # axs = axs.ravel()
            # axs[0].imshow(self.dab.img)
            # axs[0].set_title('DAB img')
            # axs[1].imshow(normed_dab)
            # axs[1].set_title('norm DAB img')
            # axs[2].imshow(tangle_activation)
            # axs[2].set_title('Tangle Activations')
            # axs[3].imshow(dab_scaled_activation.img)
            # axs[3].set_title('DAB * Tangle Activations')

            plt.show()
        # end self.logger.plots
        
        relevant_tangle_dab_msk, dab_thresh = self.process_dab(
            self.dab,
            run_normalize = True,
            scale = 1.0,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            open_r = 7, #always an odd number (7 < open_r < 13)
            close_r = 4,
            # mask = self.main_mask,
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

        # Calculate hyp annos
        # store DAB 
        tangle_activation = Frame(tangle_activation,lvl=self.dab.lvl,converter=self.dab.converter)
        tangle_activation.img = cv2.resize(tangle_activation.img, relevant_tangle_dab_msk.size(),interpolation=cv2.INTER_NEAREST)

        tangle_dab = Frame(self.dab.img[:,:,None],lvl=self.dab.lvl,converter=self.dab.converter)
        tangle_dab.img = cv2.resize(tangle_dab.img, relevant_tangle_dab_msk.size(),interpolation=cv2.INTER_NEAREST)
                
        tangle_softmax = Frame(tangle_softmax,lvl=self.dab.lvl,converter=self.dab.converter)
        tangle_softmax.img = cv2.resize(tangle_softmax.img, relevant_tangle_dab_msk.size(),interpolation=cv2.INTER_NEAREST)     
       
        # threshold probs img at tangle prob and store in a Frame
        tangle_msk = np.where(tangle_softmax.img >= softmax_prob_thresh, relevant_tangle_dab_msk.img[:,:,0], np.zeros_like(relevant_tangle_dab_msk.img[:,:,0]))
        tangle_msk = Frame(tangle_msk, lvl=self.dab.lvl, converter=self.dab.converter)

        # generate WC predictions for %AO
        wc_preds = frame_like(self.frame, (wc_probs[2] > softmax_prob_thresh).astype(np.uint8))
        wc_preds.resize(self.frame.size(), interpolation=cv2.INTER_NEAREST)

        # get the contours/polygons from the tangle mask
        tangle_msk.get_contours()
        tangle_msk.filter_contours(min_body_area=35/self.dab.converter.mpp) #default: 20/mpp
        tangle_contours = tangle_msk.get_body_contours()
        tangles = [contour.polygon for contour in tangle_contours]
        
        # get the contours/polygons from the LN mask
        # ln_msk.get_contours()
        # ln_msk.filter_contours(min_body_area=0/self.dab.converter.mpp) #default: 20/mpp
        # ln_contours = ln_msk.get_body_contours()
        # lns = [contour.polygon for contour in ln_contours]

        # tangles = []
        # # apply a coarseness mask
        # for i, tangle in enumerate(old_tangles):
        #     # - load local DAB image using bounding box of tangle polygon
        #     loc, size = tangle.bounding_box()
        #     local_dab = self.dab.copy()
        #     tangle_msk_c = tangle_msk.copy()
        #     local_dab.crop(loc,size)
        #     tangle_msk_c.crop(loc,size)
        #     # self.save_frame(odir, local_dab, 'Local_DAB_tangle_%d' %i)
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
        #     local_tangle_msk = Frame(segmentation_msk.copy().astype(np.uint8),lvl=self.dab.lvl,converter=self.dab.converter)

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
        #         axs[2].set_title('Old tangle Mask')
        #         axs[2].imshow(tangle_msk_c.img,cmap='gray')
        #         axs[3].set_title('tangle Mask')
        #         axs[3].imshow(local_tangle_msk.img)
        #         fig.tight_layout()
        #         # plt.show()

        #     # - find contours on that thresholded image surrounding the tangle polygon
        #     local_tangle_msk.get_contours()
        #     local_tangle_msk.filter_contours(min_body_area=10/self.dab.converter.mpp) #default: 20/mpp
        #     tangle_contour = local_tangle_msk.get_body_contours()
        #     new_tangles = [c.polygon for c in tangle_contour]
        #     # new_tangles = [transform_inv_poly(x, loc, params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in new_polys]
        #     tangle.translate(loc)

        #     # Plot the new tangles on the DAB and Orig img
        #     if self.logger.plots:
        #         local_frame = self.frame.copy()
        #         local_frame.crop(loc,size)
        #         fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
        #         fig.suptitle('Comparison of Orig tangle Poly vs Chan-Vese tangle Poly')
        #         axs[0].set_title('Before CV')
        #         axs[0].imshow(local_frame.img)
        #         plot_poly(axs[0],tangle,color='red')
        #         axs[1].set_title('After CV')
        #         axs[1].imshow(local_frame.img)
        #         axs[2].set_title('After EFD')
        #         axs[2].imshow(local_frame.img)
        #         for i, new_tangle in enumerate(new_tangles):
        #             plot_poly(axs[1],new_tangle,color='red')
        #         plt.show()
            
        #     # translate before saving
        #     [p.translate(-loc) for p in new_tangles]

        #     # - replace the tangle polygon w/ this contour 
        #     if new_tangles:
        #         tangles.append(new_tangles[-1])
        #
        # end of Chan-Vese loop

        #     # * done before saving the tangle annotations in run_tangle_detections

        # convert polygons to annotations and calculate confidence
        bid, antibody, region, roi_name = odir.split('/')[-4:]
        tile = roi_name.split('_')[-1]
        
        tangle_annos = []
        for tangle in tangles:
                
            # confidence is the average tangle activation within the polygon
            conf, conf_std = self.get_confidence(tangle, tangle_activation)

            # confidence is average value of the DAB intensity within the polygon
            dab_conf, dab_std = self.get_confidence(tangle, tangle_dab)

            # alpha = 0.5
            # beta = 0.5
            # plaque_probs = wc_probs[1,:,:]
            # DAB_INTENSITY = cv2.resize(self.dab.img.copy(), plaque_probs.shape, interpolation=cv2.INTER_NEAREST)
            # conf_img = Frame(alpha*DAB_INTENSITY + beta*(1-plaque_probs),lvl=self.dab.lvl,converter=self.dab.converter)
            # conf = np.mean(conf_img.img)

            # fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            # axs = axs.ravel()
            # for ax in axs:
            #     plot_poly(ax, tangle, color='red')
            # axs[0].imshow(resized_activation.img)
            # axs[0].set_title('Tangle Activation')
            # axs[1].imshow(msk.img)
            # axs[1].set_title('Confidence Mask')
            # axs[2].imshow(plaque_probs)
            # axs[2].set_title('Plaque Probs')
            # axs[3].imshow(conf_img.img)
            # axs[3].set_title('DAB Int. + Plaque Probs')
            # print('Before:',self.get_confidence(tangle,tangle_activation))
            # print('After:',np.mean(conf_img.img))
            # plt.show()

            fname = bid+'_'+antibody+'_'+region+'_'+roi_name
            tangle_anno = tangle.to_annotation(
                fname, 
                class_name='tangle detection', 
                confidence=conf,
                confidence_std=conf_std,
                dab_confidence=dab_conf,
                dab_std=dab_std,
            )
            tangle_annos.append(tangle_anno)

        # ln_annos = []
        # for ln in lns:

        #     # confidence is the average tangle activation inside the polygon
        #     conf = self.get_confidence(ln, ln_activation)

        #     fname = bid+'_'+antibody+'_'+region+'_'+tile
        #     ln_anno = ln.to_annotation(
        #         fname, class_name='LN detection', confidence=conf
        #     )
        #     ln_annos.append(ln_anno)

        tangle_poly_mask = create_mask(tangle_annos, self.frame.size(), self.frame.lvl, self.frame.converter)
        # ln_poly_mask = create_mask(ln_annos, self.frame.size(), self.frame.lvl, self.frame.converter)

        # run the %AO over predictions
        results = self.run_ao(wc_preds)
        params.data['tangle_wc_ao'] = results['ao']
        if not results['signals'] is None:
            self.save_signals(odir, results['signals']['main_deform'], 'tangle')

        results = self.run_ao(tangle_poly_mask)
        params.data['tangle_poly_ao'] = results['ao']
        if not results['signals'] is None:
            self.save_signals(odir, results['signals']['main_deform'], 'tangle_POLY')        
        
        # results = self.run_ao(ln_preds)
        # params.data['ln_wc_ao'] = results['ao']
        # if not results['signals'] is None:
        #     self.save_signals(odir, results['signals']['main_deform'], 'LN')        

        # results = self.run_ao(ln_poly_mask)
        # params.data['ln_poly_ao'] = results['ao']
        # if not results['signals'] is None:
        #     self.save_signals(odir, results['signals']['main_deform'], 'LN_POLY')

        if len(tangle_annos)>0 and self.logger.plots:
            # frame resizing 
            orig_img = cv2.resize(self.frame.img.copy(), wc_probs[2,:,:].shape, interpolation=cv2.INTER_NEAREST)
            orig_img = cv2.rotate(orig_img,cv2.ROTATE_90_CLOCKWISE)
            orig_frame = Frame(orig_img,lvl=self.frame.lvl,converter=self.frame.converter)

            # debug WC activations for each of the classes
            class_dict = {
                0: 'Background',
                1: 'GM NP',
                2: 'GM Tangle',
            }
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.ravel()
            [ax.set_axis_off() for ax in axs]
            axs[0].imshow(orig_img)
            axs[0].set_title('Orig. Frame')
            for i in range(len(axs)-1):
                axs[i+1].imshow(wc_activation[i,:,:])
                axs[i+1].set_title(class_dict[i])
            fig.suptitle('WildCat Class Activation Maps')
            plt.tight_layout()

            # Plot polygons against DAB image and activation map
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.ravel()
            [ax.set_axis_off() for ax in axs]
            fig.suptitle('Debugging Plot of a GM Tangle Detection')

            # plot the original image
            axs[0].imshow(self.frame.img) 
            axs[0].set_title('Orig. Img')

            # plot the strong DAB in the image
            axs[1].imshow(relevant_tangle_dab_msk.img)
            [plot_poly(axs[1], tangle, color='red') for tangle in tangles]
            axs[1].set_title('DAB mask')

            alpha = 0.4
            tangle_preds = frame_like(orig_frame, wc_activation[2] > softmax_prob_thresh)
            overlay = overlay_thresh(orig_frame, tangle_preds, alpha=alpha, color='red')
            overlay.img = cv2.resize(overlay.img, self.frame.size(), interpolation=cv2.INTER_NEAREST)
            axs[2].imshow(overlay.img)
            axs[2].set_title('WC Pixel Predictions')

            if self.save_images:
                self.save_frame(odir, overlay, 'TANGLE_QC')

            axs[3].imshow(cv2.resize(wc_activation[2,:,:],self.frame.size()))
            axs[3].set_title('Polygons on WC Activation Map')
            [plot_poly(axs[3], tangle, color='red') for tangle in tangles]

            plt.show()
        #
        # end of debugging
        
        tangle_annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in tangle_annos]

        # set pixel lvl to 0
        [self.dab.converter.rescale(x, 0) for x in tangle_annos]        

        # finally, write the annoations to a file
        afile = os.path.basename(self.fname).replace('.svs','.json')
        anno_fname = roi_odir+'/'+afile
        if not os.path.exists(anno_fname) or first_run:
            self.logger.debug('Writing %d annotations: %s' %(len(tangle_annos),anno_fname))
            sana_io.write_annotations(anno_fname, tangle_annos)
            first_run = False
        else:
            self.logger.debug('Appending %d annotations: %s' %(len(tangle_annos),anno_fname))
            sana_io.append_annotations(anno_fname, tangle_annos)
        # self.logger.debug('Detections written to file: %s' %anno_fname)

        # debugging
        # loads in anno file that was just written
        if len(tangle_annos) > 0 and self.logger.plots:
            slides = sana_io.read_list_file('C:/Users/eteun/dev/data/AD_tangle/lists/test.list')
            img_dir = [s for s in slides if os.path.basename(s) == os.path.basename(anno_fname).replace('.json','.svs')][0]

            # initialize loader, this will allow us to load slide data into memory
            image = sana_io.create_filepath(img_dir)

            loader = Loader(image)
            loader.set_lvl(0)

            thumb_frame = loader.load_thumbnail()
            roiname = ['Tile *']
            refclass = ['Mature Tangle']
            hypclass = ['tangle detection']

            ref_anno = sana_io.create_filepath(anno_fname, ext='.json', fpath='C:/Users/eteun/dev/data/AD_tangle/annotations/sanaz_all_v2/')

            ROIS = []
            for roi_name in roiname:
                ROIS += sana_io.read_annotations(ref_anno, name=roi_name)

            # load all reference annotations with matching class name
            REF_ANNOS = []
            for ref_class in refclass:
                # load the annotations from the annotation file
                REF_ANNOS += sana_io.read_annotations(ref_anno, class_name=ref_class)

            # load hypothesis annotations with matching class name
            HYP_ANNOS = []
            for hyp_class in hypclass:
                # load the annotations from the annotation file
                HYP_ANNOS += sana_io.read_annotations(anno_fname, class_name=hyp_class)

            # rescale the ROI annotations to the given pixel resolution
            for ROI in ROIS:
                if ROI.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(ROI, self.frame.lvl)
            for REF_ANNO in REF_ANNOS:
                if REF_ANNO.lvl != self.frame.lvl:
                    self.dab.converter.rescale(REF_ANNO, self.frame.lvl)
            for HYP_ANNO in HYP_ANNOS:
                if HYP_ANNO.lvl != self.frame.lvl:
                    self.dab.converter.rescale(HYP_ANNO, self.frame.lvl)
            
            # # create new plot w/ zoom in of ROI (self.frame)
            # fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
            # for ROI in ROIS:
            #      # Plotting the processed frame and processed frame with clusters
            #     axs[0].imshow(self.frame.img, cmap='gray')
            #     axs[0].set_title('Ref. Annos')
            #     for REF_ANNO in REF_ANNOS:
            #         # REF_ANNO.translate(params.data['loc'])
            #         plot_poly(axs[0], REF_ANNO, color='red')

            #     axs[1].imshow(self.frame.img, cmap='gray')
            #     axs[1].set_title('Hyp. Annos | Poly Count: %d' %len(HYP_ANNOS))
            #     for HYP_ANNO in HYP_ANNOS:
            #         # HYP_ANNO.translate(params.data['loc'])
            #         plot_poly(axs[1], HYP_ANNO, color='blue')


            # rescale the ROI annotations to the given pixel resolution
            for ROI in ROIS:
                if ROI.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(ROI, thumb_frame.lvl)
            for REF_ANNO in REF_ANNOS:
                if REF_ANNO.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(REF_ANNO, thumb_frame.lvl)
            for HYP_ANNO in HYP_ANNOS:
                if HYP_ANNO.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(HYP_ANNO, thumb_frame.lvl)
            
            # TODO: if you plot the raw annotations that are just rescaled to level 2 then plotting the level 2 thumbnail image should fix the left plot
            # just use loader.thumbnail as the image you're plotting
            # plot the annotations
            # get the top left coordinate and the size of the bounding centroid
            

            fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
            for ROI in ROIS:
                # Plotting the processed frame and processed frame with clusters
                axs[0].imshow(thumb_frame.img, cmap='gray')
                axs[0].set_title('Ref. Annos | Poly Count: %d' %len(REF_ANNOS))
                for REF_ANNO in REF_ANNOS:
                    # REF_ANNO.translate(params.data['loc'])
                    plot_poly(axs[0], REF_ANNO, color='red')

                axs[1].imshow(thumb_frame.img, cmap='gray')
                axs[1].set_title('Hyp. Annos | Poly Count: %d' %len(HYP_ANNOS))
                for HYP_ANNO in HYP_ANNOS:
                    # HYP_ANNO.translate(params.data['loc'])
                    plot_poly(axs[1], HYP_ANNO, color='blue')
            plt.show()
        # -------------------------------------------------------------
        # overlay the annotations onto the original frame       
    # 
    # end of run_tangle_detection

 
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
