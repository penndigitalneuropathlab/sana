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
from scipy.special import expit as sigmoid

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, plot_poly, transform_inv_poly
from sana_frame import Frame, create_mask, overlay_thresh
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from wildcat.saved_models import R13Model
from sana_loader import Loader

# debugging modules
from matplotlib import pyplot as plt

class R13Processor(HDABProcessor):
    def __init__(self, fname, frame, debug=False):
        super(R13Processor, self).__init__(fname, frame, debug)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, last_run, params, main_roi, sub_rois=[]):
        # generate the neuronal and glial severity results

        self.mask_frame(main_roi, sub_rois)

        self.run_lb_detection(odir, roi_odir, first_run, last_run, params)


        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that

        # self.manual_dab_threshold = 94

        # generate processed, thresholded DAB img mask
        # dab_img, dab_thresh = self.process_dab(self.dab,
        #     run_normalize=True,
        #     scale = 1.0,
        #     mx = 90,
        #     close_r = 0,
        #     open_r = 0,
        #     debug = False
        #     )
        # self.save_frame(odir, dab_img, 'processed_DAB')

        # generate the manually curated AO results
        # self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90)
        
        # if self.debug:
        #     fig, axs = plt.subplots(1,2)
        #     axs[0].imshow(auto_dab.img)
        #     axs[0].set_title('Auto AO DAB')
        #     axs[1].imshow(dab_img.img)
        #     axs[1].set_title('Processed DAB')
        #     fig.suptitle('Comparison of Auto AO DAB and Processed DAB')
        #     plt.show()

        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')

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

    def run_lb_detection(self, odir, roi_odir, first_run, last_run, params):
        debug = False
        last_run = False

        # if debug:
        #     fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
        #     axs = np.ravel(axs)

        model = R13Model(self.frame)
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
            lb_annos.append(lb.to_annotation(fname,class_name='LB detection',anno_name=tile,confidence=conf))


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
        
    # end of run_lb_detection

        # debugging
        # loads in anno file that was just written
        if last_run:
            slides = sana_io.read_list_file('./lists/all.list')

            img_dir = [s for s in slides if os.path.basename(s) == os.path.basename(anno_fname).replace('.json','.svs')][0]

            # initialize loader, this will allow us to load slide data into memory
            image = sana_io.create_filepath(img_dir)

            loader = Loader(image)
            loader.set_lvl(0)

            thumb_frame = loader.load_thumbnail()
            roiclass = ['Tile *']
            refclass = ['']
            hypclass = ['LB detection']

            ref_anno = sana_io.create_filepath(anno_fname, ext='.json', fpath='./annotations/')
        
            ROIS = []
            for roi_class in roiclass:
                ROIS += sana_io.read_annotations(ref_anno, name=roi_class)
            # rescale the ROI annotations to the given pixel resolution
            for ROI in ROIS:
                if ROI.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(ROI, thumb_frame.lvl)

            # load all reference annotations with matching class name
            REF_ANNOS = []
            for ref_class in refclass:
                # load the annotations from the annotation file
                REF_ANNOS += sana_io.read_annotations(ref_anno, name=ref_class)
            # rescale the ROI annotations to the given pixel resolution
            for REF_ANNO in REF_ANNOS:
                if REF_ANNO.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(REF_ANNO, thumb_frame.lvl)
            
            # load hypothesis annotations with matching class name
            HYP_ANNOS = []
            for hyp_class in hypclass:
                # load the annotations from the annotation file
                HYP_ANNOS += sana_io.read_annotations(anno_fname, class_name=hyp_class)
            # rescale the ROI annotations to the given pixel resolution
            for HYP_ANNO in HYP_ANNOS:
                if HYP_ANNO.lvl != thumb_frame.lvl:
                    self.dab.converter.rescale(HYP_ANNO, thumb_frame.lvl)
            print('HYP_ANNOS:',HYP_ANNOS)
            
            # TODO: if you plot the raw annotations that are just rescaled to level 2 then plotting the level 2 thumbnail image should fix the left plot
            # just use loader.thumbnail as the image you're plotting
            # plot the annotations
            for ROI in ROIS:
                # get the top left coordinate and the size of the bounding centroid
                loc, size = ROI.bounding_box()

                frame = loader.load_frame(loc, size)

                fig, axs = plt.subplots(1,2, sharex=True, sharey=True)

                # og_img = loader.load_thumbnail()

                # Plotting the processed frame and processed frame with clusters
                axs[0].imshow(thumb_frame.img, cmap='gray')
                for REF_ANNO in REF_ANNOS:
                    # REF_ANNO.translate(loc)
                    plot_poly(axs[0], REF_ANNO, color='red')

                axs[1].imshow(thumb_frame.img, cmap='gray')
                for HYP_ANNO in HYP_ANNOS:
                    # HYP_ANNO.translate(loc)
                    plot_poly(axs[1], HYP_ANNO, color='blue')
                plt.show()


        # if self.debug:
        #     fig, axs = plt.subplots(1,len(probs))
        #     for i in range(len(probs)):
        #         axs[0,i].imshow(probs[i], cmap='coolwarm')
        #     gs = axs[0,0].get_gridspec()
        #     for ax in axs[0:2, 0:2].flatten():
        #         ax.remove()
        #     axbig = fig.add_subplot(gs[0:2,0:2])
        #     axbig.imshow(self.frame.img)
        #     plt.show()

    #
    # end of run_lb_detection
#
# end of R13Processor

#
# end of file
