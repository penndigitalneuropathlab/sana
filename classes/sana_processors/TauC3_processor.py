
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
from sana_geo import Point, transform_inv_poly, plot_poly, transform_poly_with_params, Polygon
from sana_frame import Frame, frame_like, create_mask, overlay_thresh
from sana_loader import Loader
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.pixel_classifiers import TauC3CorticalTangleClassifier, TauC3HIPTangleClassifier

# debugging modules
from matplotlib import pyplot as plt

class TauC3Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(TauC3Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):
        self.logger.info('Running TauC3 Processor...')

        self.generate_masks(main_roi, sub_rois)

        self.main_roi = main_roi
        self.sub_rois = sub_rois

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
            softmax_prob_thresh = 0.3
            self.run_tangle_detection(odir, detection_odir, first_run, params, main_roi, softmax_prob_thresh)


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

    
    def run_tangle_detection(self, odir, roi_odir, first_run, params, main_roi, softmax_prob_thresh=0.5):   
        self.logger.info('Running Tangle detections...')
        self.logger.debug('Softmax Prob. Thresh: %0.2f' %softmax_prob_thresh)
        
        roi_class = main_roi.class_name.split('_')[1]
       
        if roi_class in ['GM']:
            self.logger.debug('Deploying Cortical TauC3 Tangle Classifier...')
            model = TauC3CorticalTangleClassifier(self.frame)
        elif roi_class in ['CA1', 'CA2', 'CA3', 'CA4', 'Subiculum', 'DG']:
            self.logger.debug('Deploying HIP TauC3 Tangle Classifier...')
            model = TauC3HIPTangleClassifier(self.frame)
        else:
            self.logger.info('No TauC3 classifier found for ROI class: %s' %roi_class)
            return
        
        tangle_idx = 2

        # decode the model to get activations of each class
        wc_activation = model.run()

        # self.logger.debug('Shape of model output: %s',str(wc_activation.shape))
        # print(self.frame.img.shape)
        wc_probs = softmax(wc_activation, axis=0)
    
        # save the output probabilities
        ofname = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_PROBS.npy'))
        with open(ofname,'wb') as f:
            np.save(f, wc_activation)

        # [2,:,:] to select Tangle predictions from model output (Background, GM NP, GM Tangle)
        tangle_activation = wc_activation[tangle_idx,:,:]
        tangle_softmax = wc_probs[tangle_idx,:,:]

        if self.logger.plots:
            # load in main roi to plot in tile
            orig_anno = main_roi.copy()
            self.frame.converter.rescale(orig_anno,self.frame.lvl)

            # make this into a function --> visual_wc_probs(class_dict)
            # debug WC activations for each of the classes
            
            class_dict = {
                0: 'Artifact',
                1: 'Background',
                2: 'GM Tangle',
            }
            
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.ravel()
            fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            axs[0].imshow(self.frame.img)
            axs[0].set_title('Orig. Frame')
            plot_poly(axs[0],orig_anno,color='red')
            for i in range(len(axs)-1):
                plot_poly(axs[i+1],orig_anno,color='red')
                axs[i+1].imshow(cv2.resize(wc_probs[i,:,:],self.frame.img[:,:,0].transpose((1,0)).shape,interpolation=cv2.INTER_NEAREST))
                axs[i+1].set_title(class_dict[i])
            fig.suptitle('WildCat Class Activation Maps')
            plt.tight_layout()
            
            plt.show()
        # end self.logger.plots
            
        # TODO: try masking DAB image by WC activations before sending to process_dab
        wc_mask = self.dab.copy()
        wc_mask.img = np.where(tangle_softmax >= softmax_prob_thresh, np.ones_like(tangle_softmax), np.zeros_like(tangle_softmax))

        dab_masked = self.dab.copy()
        wc_mask.img = wc_mask.img[:,:,None]
        dab_masked.mask(wc_mask)

        relevant_tangle_dab_msk, dab_thresh = self.process_dab(
            self.dab,
            run_normalize = True,
            scale = 1.0,
            # mn = 50,
            mx = 90, #default: 90, experiment w/ mx = 100-110
            open_r = 7, #always an odd number (7 < open_r < 13)
            close_r = 4,
            wc_mask = dab_masked,
            # reverse = False,
            # mask = self.main_mask,
            debug = self.logger.plots
        )

        
        # fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
        # fig.suptitle(main_roi.name)
        # axs[0].imshow(self.frame.img)
        # axs[0].set_title('Orig. Frame')
        # axs[1].imshow(self.dab.img)
        # axs[1].set_title('DAB Img')
        # axs[2].imshow(relevant_tangle_dab_msk.img)
        # axs[2].set_title('WC Masked DAB')
        # plt.show()
        
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
        wc_preds = frame_like(self.frame, (wc_probs[tangle_idx] > softmax_prob_thresh).astype(np.uint8))
        wc_preds.resize(self.frame.size(), interpolation=cv2.INTER_NEAREST)

        # get the contours/polygons from the tangle mask
        tangle_msk.get_contours()
        tangle_msk.filter_contours(min_body_area=35/(self.dab.converter.mpp**2)) #default: 20/mpp
        tangle_contours = tangle_msk.get_body_contours()
        tangles = [contour.polygon for contour in tangle_contours]
        
        # draw square polygons around WC activations
        wc_mask = np.where(tangle_softmax.img >= softmax_prob_thresh, np.ones_like(tangle_softmax.img), np.zeros_like(tangle_softmax.img)).astype('uint8')
        wc_mask = Frame(wc_mask, lvl=self.dab.lvl, converter=self.dab.converter)


        # fig, axs = plt.subplots(1,1)
        # axs.imshow(wc_mask.img)
        # axs.set_title('WC Mask')

        wc_mask.get_contours()
        wc_mask.filter_contours(min_body_area=35/(self.dab.converter.mpp**2)) #default: 20/mpp
        wc_contours = wc_mask.get_body_contours()
        wc_polygons = [c.polygon.connect() for c in wc_contours]

        # frame_copy = frame_like(self.frame,np.zeros_like(self.frame.img))
        # [cv2.rectangle(frame_copy.img,(c0[0],c0[1]),(c1[0],c1[1]),(255,255,255),1) for c0,c1 in coords]
        # wc_polygons = [poly.connect() for poly in wc_polygons]
        # [plot_poly(axs,poly,color='red') for poly in wc_polygons]
        # plt.show()
        # exit()

        # box_polys = [Polygon((c0[0],c0[1]),(c1[0],c1[1])) for c0,c1 in coords]
        # print(coords[0])
        # box_polys = []
        # for c in coords:
        #     x = c[0][0]
        #     y = c[0][1]
        #     w = c[1][0]
        #     h = c[1][1]
        #     # print(x,y,w,h)
        #     box_polys.append(Polygon((x,w,w,x,x),(y,y,h,h,y),lvl=self.frame.lvl,is_micron=False))

        # fig, axs = plt.subplots(1,3)
        # axs[0].imshow(self.frame.img)
        # axs[1].imshow(frame_copy.img)
        # axs[2].imshow(self.frame.img)
        # # plot_poly(axs[2],test_poly,color='red')
        # [plot_poly(axs[2],poly,color='red') for poly in box_polys]
        # plt.show()

        # convert polygons to annotations and calculate confidence
        bid, antibody, region, roi_name = odir.split('/')[-4:]
        tile = roi_name.split('_')[-1]    
        
        # convert WC polygons to annotations
        wc_annos = []
        for poly in wc_polygons:
            fname = bid+'_'+antibody+'_'+region+'_'+roi_name
            wc_anno = poly.to_annotation(
                fname, 
                anno_name=tile,
                class_name='wc detection'
            )
            wc_annos.append(wc_anno)

        
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


            fname = bid+'_'+antibody+'_'+region+'_'+roi_name
            tangle_anno = tangle.to_annotation(
                fname, 
                anno_name=tile,
                class_name='tangle detection', 
                confidence=conf,
                confidence_std=conf_std,
                dab_confidence=dab_conf,
                dab_std=dab_std,
            )
            tangle_annos.append(tangle_anno)

        tangle_poly_mask = create_mask([a for a in tangle_annos if a.confidence >= 1.31], self.frame.size(), self.frame.lvl, self.frame.converter)
        # run the %AO over predictions
        if np.max(wc_preds.img)>0:
            results = self.run_ao(wc_preds)
            params.data['tangle_wc_ao'] = results['ao']
            if not results['signals'] is None:
                self.save_signals(odir, results['signals']['main_deform'], 'tangle')
        else:
            params.data['tangle_wc_ao'] = 0

        if len(tangle_annos) > 0 and np.max(tangle_poly_mask.img)>0:
            results = self.run_ao(tangle_poly_mask)
            params.data['tangle_poly_ao'] = results['ao']
            if not results['signals'] is None:
                self.save_signals(odir, results['signals']['main_deform'], 'tangle_POLY')        

            # self.logger.debug('%AO:')
            # self.logger.debug('WC Tangle %AO:',params.data['tangle_wc_ao'])
            # self.logger.debug('Polygon Tangle %AO:',params.data['tangle_poly_ao'])
            # self.logger.debug()
        else:
            params.data['tangle_poly_ao'] = 0

        params.data['poly_count'] = len(tangle_annos)

        if len(tangle_annos)>0 and self.save_images:
            # frame resizing 
            orig_img = cv2.resize(self.frame.img.copy(), wc_probs[tangle_idx,:,:].transpose((1,0)).shape, interpolation=cv2.INTER_NEAREST)
            orig_frame = Frame(orig_img,lvl=self.frame.lvl,converter=self.frame.converter)
            tangle_preds = frame_like(orig_frame, wc_activation[tangle_idx,:,:] > softmax_prob_thresh)
            overlay = overlay_thresh(orig_frame, tangle_preds, alpha=0.3, color='red')
            overlay.img = cv2.resize(overlay.img, self.frame.size(), interpolation=cv2.INTER_NEAREST)
            
            fig, axs = plt.subplots(1,1)
            tile = odir.split('_')[-1]
            fig.suptitle(os.path.basename(self.fname).replace('.svs','')+' | '+tile+'\n'+os.path.basename(main_roi.file_name))
            axs.set_axis_off()
            axs.imshow(overlay.img)
            roi_anno = main_roi.copy()
            overlay.converter.rescale(roi_anno,overlay.lvl)
            plot_poly(axs,roi_anno,color='red',linewidth=0.75)
            [plot_poly(axs,tangle,color='blue',linewidth=0.5) for tangle in tangle_annos if tangle.inside(roi_anno)]
            plt.savefig(os.path.join(odir,'%s_TANGLE_ANNO.png') %os.path.basename(self.fname).replace('.svs',''),dpi=500)
            plt.close()


            self.save_frame(odir, overlay, 'TANGLE_QC')
            self.logger.debug('TANGLE_QC image saved: %s' %odir)
            # orig_anno = main_roi.copy()
            # overlay.converter.rescale(orig_anno,overlay.lvl)

            # # use this to plot ROI polygon onto overlay image
            # fig, axs = plt.subplots(1,1)
            # axs.set_axis_off()
            # axs.imshow(overlay.img)
            # plot_poly(axs,orig_anno,color='red',linewidth=0.6)
            # plt.savefig(os.path.join(odir,'%s_anno_heatmap.png') %os.path.basename(self.fname).replace('.svs',''),dpi=500)



        if len(tangle_annos)>0 and self.logger.plots:
            # frame resizing 
            orig_img = cv2.resize(self.frame.img.copy(), wc_probs[tangle_idx,:,:].transpose((1,0)).shape, interpolation=cv2.INTER_NEAREST)
            orig_frame = Frame(orig_img,lvl=self.frame.lvl,converter=self.frame.converter)

            # debug WC activations for each of the classes
            class_dict = {
                0: 'Artifact',
                1: 'Background',
                2: 'Tangle'
            }
            
            fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.ravel()
            fig.suptitle('Orig. Frame to Compare WildCat Activation Maps')
            axs[0].imshow(self.frame.img)
            axs[0].set_title('Orig. Frame')
            plot_poly(axs[0],orig_anno,color='red')
            for i in range(len(axs)-1):
                plot_poly(axs[i+1],orig_anno,color='red')
                axs[i+1].imshow(cv2.resize(wc_probs[i,:,:],self.frame.img[:,:,0].transpose((1,0)).shape,interpolation=cv2.INTER_NEAREST))
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
            tangle_preds = frame_like(orig_frame, wc_probs[tangle_idx,:,:] > softmax_prob_thresh)
            overlay = overlay_thresh(orig_frame, tangle_preds, alpha=alpha, color='red')
            overlay.img = cv2.resize(overlay.img, self.frame.size(), interpolation=cv2.INTER_NEAREST)
            axs[2].imshow(overlay.img)
            axs[2].set_title('WC Pixel Predictions')

            axs[3].imshow(cv2.resize(wc_activation[tangle_idx,:,:],self.frame.size()))
            axs[3].set_title('Polygons on WC Activation Map')
            [plot_poly(axs[3], tangle, color='red') for tangle in tangles]

            if self.save_images:
                orig_anno = main_roi.copy()
                overlay.converter.rescale(orig_anno,overlay.lvl)

                # fig, axs = plt.subplots(1,1)
                # axs.set_axis_off()
                # axs.imshow(overlay.img)
                # plot_poly(axs,orig_anno,color='red',linewidth=0.6)
                # [plot_poly(axs,tangle,color='blue') for tangle in tangle_annos]
                # plt.savefig(os.path.join(odir,'%s_anno_heatmap.png') %os.path.basename(self.fname).replace('.svs',''),dpi=500)
                self.save_frame(odir, overlay, 'TANGLE_QC')

            fig, axs = plt.subplots(1,1)
            axs.set_axis_off()
            fig.suptitle('Final SANA Polygons')
            axs.imshow(self.frame.img)
            plot_poly(axs,main_roi,color='red')
            [plot_poly(axs,tangle,color='red') for tangle in tangle_annos]

            plt.show()
        #
        # end of debugging
        
        wc_annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in wc_annos]
        tangle_annos = [transform_inv_poly(x, params.data['loc'], params.data['crop_loc'], params.data['M1'], params.data['M2']) for x in tangle_annos]

        # set pixel lvl to 0
        [self.dab.converter.rescale(x, 0) for x in wc_annos]        
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
        
        # write WC box polygons to file
        os.makedirs(os.path.join(roi_odir,'WC_ANNOS'),exist_ok=True)
        afile = os.path.basename(self.fname).replace('.svs','_WC_ANNOS.json')
        anno_fname = os.path.join(roi_odir,'WC_ANNOS',afile)
        if not os.path.exists(anno_fname) or first_run:
            self.logger.debug('Writing %d WC box polygons: %s' %(len(wc_annos),anno_fname))
            sana_io.write_annotations(anno_fname, wc_annos)
            first_run = False
        else:
            self.logger.debug('Appending %d WC box polygons: %s' %(len(wc_annos),anno_fname))
            sana_io.append_annotations(anno_fname, wc_annos)
        self.logger.debug('Detections written to file: %s' %anno_fname)

        # debugging
        # loads in anno file that was just written out
        # if self.logger.plots and len(tangle_annos)>0:
        if self.logger.plots:
            slides = sana_io.read_list_file('C:/Users/eteun/dev/data/TauC3_tangles/lists/TauC3_HIP_test.list')
            img_dir = [s for s in slides if os.path.basename(s) == os.path.basename(anno_fname).replace('.json','.svs')][0]

            # initialize loader, this will allow us to load slide data into memory
            image = sana_io.create_filepath(img_dir)

            loader = Loader(image)
            loader.set_lvl(0)

            thumb_frame = loader.load_thumbnail()
            roiname = ['Tile *']
            refclass = ['tangle']
            hypclass = ['tangle detection']

            ref_anno = sana_io.create_filepath(anno_fname, ext='.json', fpath='C:/Users/eteun/dev/data/TauC3_tangles/annotations/TauC3_all_process')

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

#
# end of TauC3Processor

#
# end of file
