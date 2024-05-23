import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# custom modules
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import plot_poly
from sana_frame import Frame, overlay_thresh, mean_normalize, create_mask
import sana_io


class SMI32Processor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(SMI32Processor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    # TODO: might not even need run?
    def run(self, odir, detection_odir, first_run, params, main_roi, sub_rois=[]):

        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.4 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 99
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        # NOTE: this value is a little strict to remove some background
        # TODO: do we want to run some opening filter?
        self.run_auto_ao(odir, params, scale=0.7)

        self.run_somas(odir, params, main_roi)
                
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run    

    def run_somas(self,odir,params,main_roi):
        soma_odir = os.path.join(odir,'somas')
        self.run_auto_soma_ao(soma_odir, params, scale=0.7, mx=90, open_r=5, close_r=10)
        # filtered_frame = self.filter_dendrites(h=31,w=11)

        fig, axs = plt.subplots(1,4,figsize=(15,5),sharex=True,sharey=True)
        axs = axs.ravel()

        axs[0].set_title('Original Image')
        axs[0].imshow(self.frame.img)
        plot_poly(axs[0],main_roi,color='black')
        
        axs[1].set_title('DAB Image')
        axs[1].imshow(self.dab.img)

        # axs[2].set_title('Morph. DAB Img')
        # axs[2].imshow(self.soma_dab_thresh_img.img)


        self.soma_dab_thresh_img.get_contours()
        self.soma_dab_thresh_img.filter_contours(min_body_area=75/(self.dab.converter.mpp**2)) #default: 20/mpp
        neuron_contours = self.soma_dab_thresh_img.get_body_contours()
        neurons = [contour.polygon for contour in neuron_contours]
        print('Number of neurons:',len(neurons))

        # neuron_areas = [n.area()*(self.dab.converter.mpp**2) for n in neurons]
       
        # neuron_circs = [n.circularity() for n in neurons]

        # neuron_eccs = [n.eccentricity() for n in neurons]

        # nbins = np.arange(np.nanmin(neuron_areas),np.nanmax(neuron_areas),20)
        # fig, hist_ax = plt.subplots(3,1)
        # hist_ax = hist_ax.ravel()
        # n, bins, patches = hist_ax[0].hist(neuron_areas,bins='auto',rwidth=0.8)
        # hist_ax[0].set_title('Neuron Areas')
        # n, bins, patches = hist_ax[1].hist(neuron_circs,bins='auto',rwidth=0.8)
        # hist_ax[1].set_title('Neuron Circularity')
        # n, bins, patches = hist_ax[2].hist(neuron_eccs,bins='auto',rwidth=0.8)
        # hist_ax[2].set_title('Neuron Eccentricity')

        cleaned_neurons = [n for n in neurons if n.circularity()>0.05 and n.eccentricity()<0.95]

        axs[2].set_title('%AO')
        axs[2].imshow(self.auto_overlay.img)

        # axs[3].set_title('Cleaned Neuron Mask')
        # cleaned_neuron_mask = create_mask(cleaned_neurons,self.frame.size(),self.frame.lvl,self.frame.converter)
        # axs[3].imshow(cleaned_neuron_mask.img)
        
        axs[3].set_title('Neuron %AO')
        # [plot_poly(axs[5],n,color='red',linewidth=0.8) for n in neurons]
        # [plot_poly(axs[5],n,color='green',linewidth=0.8) for n in cleaned_neurons]
        # clean_overlay = overlay_thresh(
        #     self.frame, create_mask(neurons,self.frame.size(),self.frame.lvl,self.frame.converter), 
        #     color='red', alpha=0.6,
        #     main_mask=self.main_mask, sub_masks=self.sub_masks,
        #     main_roi=self.main_roi, sub_rois=self.sub_rois)
        
        clean_overlay = overlay_thresh(
            self.frame, create_mask(cleaned_neurons,self.frame.size(),self.frame.lvl,self.frame.converter),
            color='green', alpha=0.8,
            main_mask=self.main_mask, sub_masks=self.sub_masks,
            main_roi=self.main_roi, sub_rois=self.sub_rois)
        axs[3].imshow(clean_overlay.img)

        # cleaned_neuron_overlay = overlay_thresh(
        #         self.frame, cleaned_neuron_mask,
        #         main_mask=self.main_mask, sub_masks=self.sub_masks,
        #         main_roi=self.main_roi, sub_rois=self.sub_rois)
        # axs[5].imshow(cleaned_neuron_overlay.img)


        plt.tight_layout()
        for a in axs:
            a.axis('off')

        # soma_img_fname = os.path.join(odir,'neuron_soma_grant_img.png')
        # plt.savefig(soma_img_fname,dpi=500)
        plt.show()
    #
    # end of run_somas

    def filter_dendrites(self, h, w):
        mid = h//2
        kern = np.zeros((h,h))
        kern[:,mid-w//2:mid+w//2] = 1
        frame_img = cv2.morphologyEx(self.frame.img, cv2.MORPH_OPEN, kern)
        return Frame(frame_img,lvl=self.frame,converter=self.frame.converter)
    #
    # end of filter_dendrites

    
    # performs normalization, smoothing, and histogram
    def run_auto_soma_ao(self, odir, params, scale=1.0, mx=255, open_r=0, close_r=0, min_background=0):
        self.soma_dab_thresh_img, self.auto_dab_threshold = \
            self.process_dab_soma(
                self.dab,
                run_normalize = True,
                min_background = 0,
                scale=scale,
                mx=mx,
                close_r=close_r,
                open_r=open_r,
                debug = self.logger.plots,
            )

        # run the AO process
        results = self.run_ao(self.auto_dab_thresh_img)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']
        params.data['auto_ao'] = results['ao']
        params.data['auto_sub_aos'] = results['sub_aos']
        params.data['auto_stain_threshold'] = self.auto_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir,'')

        # write params data to csv
        self.save_params(odir,params)

        # save the feature signals
        # signals = results['signals']
        # if signals:
        #     self.save_signals(odir, signals['normal'], 'AUTO_NORMAL')
        #     self.save_signals(odir, signals['main_deform'], 'AUTO_MAIN_DEFORM')
        #     if 'sub_deform' in signals:
        #         self.save_signals(odir, signals['sub_deform'], 'AUTO_SUB_DEFORM')
        
        if self.save_images:
            # save the images used in processing
            self.soma_overlay = overlay_thresh(
                self.frame, self.soma_dab_thresh_img, alpha=0.6,
                main_mask=self.main_mask, sub_masks=self.sub_masks,
                main_roi=self.main_roi, sub_rois=self.sub_rois)
            self.save_frame(odir, self.soma_dab_thresh_img, 'SOMA_THRESH')
            self.save_frame(odir, self.soma_overlay, 'SOMA_QC')

        # save the compressed boolean array during slidescan mode
        # if self.roi_type == 'SLIDESCAN':
        #     self.save_ao_arr(odir, self.auto_dab_thresh_img)
    #
    # end of run_auto_soma_ao

    # function takes in a DAB Frame object, extracting a DAB thresholded image
    def process_dab_soma(self, frame, run_normalize=False, scale=1.0, mn=0, mx=255, 
                    close_r=0, open_r=0, min_background=0, wc_mask=None, mask=None, debug=False):
        self.logger.info('Processing DAB...')
        if debug and run_normalize:
            fig, axs = plt.subplots(2,3, sharex=True,sharey=True)
            axs = axs.ravel()
        elif debug and not run_normalize:
            fig, axs = plt.subplots(2,2, sharex=True,sharey=True)
            axs = axs.ravel()


        dab = frame.copy()
        # plot #1
        if debug:
            axs[0].imshow(self.frame.img)
            axs[0].set_title('Orig. Img')

        # plot #2
        if debug:
            axs[1].imshow(dab.img)
            axs[1].set_title('DAB Img')
            
        if run_normalize:
            # normalize the image
            # TODO: rename mean_normalize --> bckgrnd subtraction (denoising)
            self.soma_dab_norm = mean_normalize(dab, min_background=min_background, debug=False)

            # TODO: run anisodiff
            # smooth the image
            self.soma_dab_norm.anisodiff()

            # self.dab_norm.img = self.dab_norm.img[:,:,None]

            dab = self.soma_dab_norm

            # plot #3
            if debug:
                axs[2].imshow(dab.img)
                axs[2].set_title('Normalized/Smoothed DAB Img')

        # get the histogram of only valid data
        masked_dab = dab.copy()
        masked_dab.mask(self.main_mask)
        masked_dab.mask(self.ignore_mask)
        if not wc_mask is None:
            masked_dab.mask(wc_mask)

            if debug:
                fig1, axs1 = plt.subplots(1,2)
                axs1[0].imshow(dab.img)
                axs1[0].set_title('Norm/Smooth DAB')
                axs1[1].imshow(masked_dab.img)
                axs1[1].set_title('Masked DAB')
        
        dab = masked_dab


        dab_hist = masked_dab.histogram()

        # get the stain threshold
        dab_threshold = max_dev(dab_hist, scale=scale, mx=mx, debug=debug, show_debug=False)

        if dab_threshold < mn:
            dab_threshold = mn

        # apply the thresholding
        dab.img = np.where(dab.img < dab_threshold, 0, 1).astype(np.uint8)

        self.dab_thresh_soma = dab.copy()
        # fig1, ax1 = plt.subplots(1,1)
        # ax1.imshow(dab.img)
        # ax1.axis('off')
        # fig1.savefig(r'C:/Users/eteun/dev/data/AT8_tangles/thresholded_img.png',dpi=500)
        # plt.close()
            
        # plot #2 or #4
        if debug:
            axs[-3].imshow(dab.img)
            axs[-3].set_title('Thresholded Img')

        if close_r > 0:
            close_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_r, close_r))
            dab.img = cv2.morphologyEx(dab.img, cv2.MORPH_CLOSE, close_kern)
        if open_r > 0:
            open_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (open_r, open_r))
            dab.img = cv2.morphologyEx(dab.img, cv2.MORPH_OPEN, open_kern)

        # plot #3 or #5
        if debug:
            axs[-2].imshow(dab.img)
            axs[-2].set_title('Morph. Filter of Thresholded Img')
        

        img_final = dab.copy().img

        # plot #4 or #6
        if debug:
            axs[-1].imshow(img_final)
            axs[-1].set_title('Final Processed Image')
            fig.suptitle('Debugging Plots for DAB Processing\n'+
                        'DAB Threshold: %d' %dab_threshold)
            plt.tight_layout()
            # plt.show()

        return Frame(img_final, lvl=dab.lvl, converter=dab.converter), dab_threshold
    #
    # end of process_dab_soma


    
#
# end of SMI32Processor

#
# end of file
