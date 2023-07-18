
# installed modules
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, overlay_thresh, create_mask
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_antibody_processor import Processor
from sana_filters import minmax_filter
from sana_heatmap import Heatmap
from sana_geo import Point, Polygon, transform_inv_poly, hull_to_poly, Line, transform_poly_with_params

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly

# generic Processor for H-DAB stained slides
# performs stain separation and rescales the data to 8 bit pixels
class HDABProcessor(Processor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(HDABProcessor, self).__init__(fname, frame, logger, **kwargs)

        # prepare the stain separator
        self.ss = StainSeparator('H-DAB', self.stain_vector)

        # if self.logger.plots:
        #     ds = 20
        #     img = self.frame.img
        #     img = img[::ds, ::ds].astype(float) / 255
        #     odimg = np.clip(-np.log10(img), 0, None)
        #     odimg[odimg==np.inf] = 0
        #     r, g, b = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
        #     odr, odg, odb = odimg[:,:,0].flatten(), odimg[:,:,1].flatten(), odimg[:,:,2].flatten()
        #     colors = [(r[i], g[i], b[i]) for i in range(len(r))]
        #     fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='3d'))
        #     ax.scatter(odr, odg, odb, color=colors)
        #     x, y, z = self.ss.stain_vector.orig_v[0]
        #     ax.quiver(0, 0, 0, x, y, z, color=(x, y, z))
        #     x, y, z = self.ss.stain_vector.orig_v[1]
        #     ax.quiver(0, 0, 0, x, y, z, color=(x, y, z))
        #     fig.suptitle('HDABProcessor | Stains and Stain Vectors')
        #     plt.show()
        #     # exit()

        # separate out the HEM and DAB stains
        self.stains = self.ss.run(self.frame.img)
        self.hem = Frame(self.stains[:,:,0], frame.lvl, frame.converter)
        self.dab = Frame(self.stains[:,:,1], frame.lvl, frame.converter)
        self.counter = Frame(self.stains[:,:,0]-self.stains[:,:,1], frame.lvl, frame.converter)

        # rescale the OD stains to 8 bit pixel values
        # NOTE: this uses the physical min/max of the stains based
        #       on the stain vector used
        # TODO: this compresses teh digital space, need to scale it back to 0 and 255!
        # self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        # calculate the manual dab threshold if a qupath threshold was given
        if self.qupath_threshold:
            self.manual_dab_threshold = \
                255*(self.qupath_threshold - self.ss.min_od[1]) / \
                (self.ss.max_od[1] - self.ss.min_od[1])

        self.gray = self.frame.copy()
        self.gray.to_gray()
        self.hem_gray = self.hem.copy()
        self.hem_gray.to_gray()
        self.dab_gray = self.dab.copy()
        self.dab_gray.to_gray()
    #
    # end of constructor

    def run(self, odir, params, scale, mx, open_r, close_r, min_background, **kwargs):
        super().run(odir, params, **kwargs)
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=scale, mx=mx, open_r=open_r, close_r=close_r, min_background=min_background)

        # detect the cells in the HEM counterstain
        if self.run_cells:
            self.run_cell_detection(odir, params)

        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run
    
    # performs a simple threshold using a manually selected cut off point
    # then runs the %AO process
    def run_manual_ao(self, odir, params):

        # apply the thresholding
        self.manual_dab_thresh = self.dab.copy()
        self.manual_dab_thresh.threshold(self.manual_dab_threshold, 0, 255)

        # TODO: the range of this function does not make sense? getting >100 for %AO
        results = self.run_ao(self.manual_dab_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']
        params.data['manual_ao'] = results['ao']
        params.data['manual_sub_aos'] = results['sub_aos']
        params.data['manual_stain_threshold'] = self.manual_dab_threshold

        # write params data to csv
        self.save_params(odir, params)
        
        # save the images used in processing
        if self.save_images:
            self.manual_overlay = overlay_thresh(
                self.frame, self.manual_dab_thresh,
                main_mask=self.main_mask, sub_masks=self.sub_masks,
                main_roi=self.main_roi, sub_rois=self.sub_rois)
            self.save_frame(odir, self.manual_dab_thresh, 'MANUAL_THRESH')
            self.save_frame(odir, self.manual_overlay, 'MANUAL_QC')

        # save the feature signals
        signals = results['signals']        
        if signals:
            self.save_signals(odir, signals['normal'], 'MANUAL_NORMAL')
            self.save_signals(odir, signals['main_deform'], 'MANUAL_MAIN_DEFORM')
            if 'sub_deform' in signals:
                self.save_signals(odir, signals['sub_deform'], 'MANUAL_SUB_DEFORM')
    #
    # end of run_manual_ao

    # TODO: rename scale/max
    # function takes in a DAB Frame object, extracting a DAB thresholded image
    def process_dab(self, frame, run_normalize=False, scale=1.0, mx=255, close_r=0, open_r=0, min_background=0, mask = None, debug=False):
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
            dab = mean_normalize(dab, min_background=min_background, debug=debug)

            # TODO: run anisodiff
            # smooth the image
            dab.anisodiff()

            # plot #3
            if debug:
                axs[2].imshow(dab.img)
                axs[2].set_title('Normalized/Smoothed DAB Img')

        # get the histogram of only valid data
        masked_dab = dab.copy()
        masked_dab.mask(self.main_mask)
        masked_dab.mask(self.ignore_mask)
        dab_hist = masked_dab.histogram()

        # get the stain threshold
        dab_threshold = max_dev(dab_hist, scale=scale, mx=mx, debug=debug, show_debug=False)

        # apply the thresholding
        dab.img = np.where(dab.img < dab_threshold, 0, 1).astype(np.uint8)

        if not mask is None:
            dab.mask(mask)

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


        # plot #4 or #6
        img_final = ((dab.img != 0) & (dab_threshold != 0)).astype(np.uint8)

        if debug:
            axs[-1].imshow(img_final)
            axs[-1].set_title('Final Processed Image')
            fig.suptitle('Debugging Plots for DAB Processing\n'+
                        'DAB Threshold: %d' %dab_threshold)
            plt.tight_layout()
            plt.show()

        return Frame(img_final, lvl=dab.lvl, converter=dab.converter), dab_threshold

    # performs normalization, smoothing, and histogram
    # TODO: rename scale to something better
    # TODO: add switches to turn off/on mean_norm, anisodiff, morph
    def run_auto_ao(self, odir, params, scale=1.0, mx=255, open_r=0, close_r=0, min_background=0):
        self.auto_dab_thresh_img, self.auto_dab_threshold = \
            self.process_dab(
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
        signals = results['signals']
        if signals:
            self.save_signals(odir, signals['normal'], 'AUTO_NORMAL')
            self.save_signals(odir, signals['main_deform'], 'AUTO_MAIN_DEFORM')
            if 'sub_deform' in signals:
                self.save_signals(odir, signals['sub_deform'], 'AUTO_SUB_DEFORM')
        
        if self.save_images:
            # save the images used in processing
            self.auto_overlay = overlay_thresh(
                self.frame, self.auto_dab_thresh_img,
                main_mask=self.main_mask, sub_masks=self.sub_masks,
                main_roi=self.main_roi, sub_rois=self.sub_rois)
            self.save_frame(odir, self.auto_dab_thresh_img, 'AUTO_THRESH')
            self.save_frame(odir, self.auto_overlay, 'AUTO_QC')

        # save the compressed boolean array during slidescan mode
        if self.roi_type == 'SLIDESCAN':
            self.save_ao_arr(odir, self.auto_dab_thresh_img)
    #
    # end of run_auto_ao

    # TODO: comment and describe parameters, why current values are selected!
    def run_segment(self, odir, params, landmarks, padding=0):

        hem = self.hem.copy()
        hem.rescale(0, 1) # HOTFIX: included to allow better thresholding
        hem.anisodiff() # HOTFIX: this is most likely needed
        
        # detect the the hematoxylin cells
        # TODO: try hem - dab??? some faint HEM comes from the dark DAB
        # TODO: try 2 iterations
        disk_r = 5 # HOTFIX: anything smaller and we'll detect everything
        sigma = 5 # HOTFIX: helps to smooth interior of cells, but worried about borders
        n_iterations = 1 # HOTFIX: >1 does not work
        close_r = 5 # HOTFIX: 9 was used previously for dendrites in NeuN
        open_r = 7 
        clean_r = 9 # HOTFIX: this helps to remove vessels

        hem_hist = hem.histogram()
        
        # HOTFIX: first get the csf threshold, this gets a better HEM threshold, NOTE not tested yet on larger data
        #csf_threshold = max_dev(hem_hist, scale=1.0, mx=50, debug=self.logger.plots)

        # HOTFIX V2: just remove all low data, this might be more stable since the above line doesn't always find the true csf threshold
        csf_threshold = 10

        # TODO: need a test to make sure the csf threshold was correctly identified, maybe look for CSF in the original image?
        hem_hist[:csf_threshold] = 0

        # HOTFIX: this needs to be tested on large data
        scale = 1.0
        mx = 100     
        hem_threshold = max_dev(hem_hist, scale=scale, mx=mx, debug=self.logger.plots)
        
        self.main_mask = None
        hem_cells = self.segment_cells(
            hem, hem_threshold, disk_r, sigma, n_iterations,
            close_r, open_r, clean_r)

        # write the cell segmentations
        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='CELLS', fpath=odir)
        hem_annos = [x.to_annotation(ofname, 'HEMCELL') for x in hem_cells]
        [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in hem_annos]
        sana_io.write_annotations(ofname, hem_annos)

        # tile size and step for the convolution operation to calculate feats
        tsize = Point(400, 300, True)
        tstep = Point(20, 20, True) # HOTFIX: 15 was way too low!

        # calculate features from the cell detections
        # TODO: include other pixel features other than intensity
        heatmap = Heatmap(self.hem, hem_cells, tsize, tstep, min_area=0)
        #feats = heatmap.run([heatmap.density, heatmap.intensity, heatmap.area])
        #feats_labels = ['DENSITY', 'INTENSITY', 'AREA']
        feats = heatmap.run([heatmap.density, heatmap.area])        
        feats_labels = ['DENSITY', 'AREA']

        # standardize the feats so that they all contribute equally to the fitness calculation
        feats[feats==np.nan] = 0
        mu = np.mean(feats, axis=(1,2))[:,None,None]
        sigma = np.std(feats, axis=(1,2))[:,None,None]
        sigma[sigma==0] = np.inf
        feats = (feats - mu) / sigma

        # calculate the grayscale intensity throughout the image
        # TODO: is this the best feat?
        self.gray = self.frame.copy()
        self.gray.to_gray()
        gray = cv2.GaussianBlur(self.gray.img, ksize=(0,0), sigmaX=40, sigmaY=1)
        gray_ds = np.array([30.0, 10.0])
        dsize = (int(round(gray.shape[1]/gray_ds[0])), int(round(gray.shape[0]/gray_ds[1])))
        gray = cv2.resize(gray, dsize)
        gray = gray[None, :, :]
        
        # detect the CSF to GM boundary with the grayscale intensity
        # NOTE: using the CSF and WM landmarks since the GM landmark grayscale is pretty dark and it errors
        #       on the layer1/2 boundary sometimes
        # TODO: make this a incline not step!
        self.logger.info('Fitting CSF Boundary')
        landmarks[0,1] = 0.0

        # detect the CSF to GM boundary with the grayscale feature
        csf_gm_score = self.get_score_matrix(
            gray[0], gray_ds[1], landmarks[0,1], landmarks[1,1])
        csf_gm = np.argmax(csf_gm_score, axis=0)
        
        # detect the GM to WM boundary with the cell features
        feats_ds = heatmap.tiler.ds
        self.logger.info('Fitting GM Boundary')        
        gm_wm_score, feat_ind = self.get_score_matrix_on_feat(
            feats, heatmap.tiler.ds[1], landmarks[1,1], landmarks[2,1])
        gm_wm = np.argmax(gm_wm_score, axis=0)
        
        # get the x arrays for the boundaries
        csf_gm_x = np.linspace(0, feats.shape[2], csf_gm.shape[0])
        gm_wm_x = np.linspace(0, feats.shape[2], gm_wm.shape[0])
        
        # interpolate so that the boundaries are equal in length
        feats_padding = self.frame.padding / feats_ds[0]
        new_x = np.linspace(feats_padding, feats.shape[2]-feats_padding, 100)
        csf_gm_f = interp1d(csf_gm_x, csf_gm)
        gm_wm_f = interp1d(gm_wm_x, gm_wm)
        csf_gm_y = csf_gm_f(new_x) * gray_ds[1] / feats_ds[1]
        gm_wm_y = gm_wm_f(new_x)

        if self.logger.plots:
            fig, axs = plt.subplots(1, feats.shape[0]+3)
            axs[0].imshow(self.frame.img)
            axs[1].imshow(self.hem.img)
            axs[2].imshow(gray[0])
            axs[2].set_aspect(self.frame.img.shape[1] / self.frame.img.shape[0])
            for i in range(feats.shape[0]):
                axs[i+3].imshow(feats[i])
                axs[i+3].set_title(feats_labels[i])
                axs[i+3].plot(new_x, csf_gm_y, color='red')
                axs[i+3].plot(new_x, gm_wm_y, color='blue')
                axs[i+3].plot(landmarks[:,0]/feats_ds[0], landmarks[:,1]/feats_ds[1], 'x', color='purple')

        # generate the Line annotations
        csf_gm = Line(new_x, csf_gm_y, False, self.frame.lvl).to_annotation(ofname, 'CSF_BOUNDARY', connect=False)
        gm_wm = Line(new_x, gm_wm_y, False, self.frame.lvl).to_annotation(ofname, 'GM_WM_BOUNDARY', connect=False)

        # scale back to original resolution
        csf_gm[:,0] *= feats_ds[0]
        gm_wm[:,0] *= feats_ds[0]
        csf_gm[:,1] *= feats_ds[1]
        gm_wm[:,1] *= feats_ds[1]

        if self.logger.plots:
            fig, ax = plt.subplots(1,1)
            ax.imshow(self.frame.img)
            plot_poly(ax, csf_gm, color='black')
            plot_poly(ax, gm_wm, color='black')            
            plt.show()

        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='GM', fpath=odir)
        boundaries = [csf_gm, gm_wm]
        
        # inverse transform the annotations
        [transform_poly_with_params(x, params, inverse=True) for x in boundaries]

        # finally, write the predicted boundaries
        sana_io.write_annotations(ofname, boundaries)

        # save the feature images
        for i in range(feats.shape[0]):
            self.save_array(odir, feats[i], feats_labels[i])
        self.save_array(odir, gray, 'GRAY')
        
        # save all parameters used to load and transform the frame
        self.save_params(odir, params)            
    #
    # end of run_segment

    # creates a simple step function
    # TODO: make more sophisticated templates
    def get_step_template(self, N, v0, v1, x0):
        template = np.zeros(N)
        template[:x0] = v0
        template[x0:] = v1
        return template
    #
    # end of get_step_template
    
    # gets the correlation score between the feat signal and various templates
    def get_boundary_scores(self, sig, v0, v1, pad_st, pad_en):

        # for stability, don't start at 0
        mi_x0 = 2
        score = np.zeros(sig.shape[0])

        # loop through all possible boundary locations
        for x0 in range(mi_x0, score.shape[0]-mi_x0):

            # get the template for this boundary decision
            template = self.get_step_template(sig.shape[0], v0, v1, x0)

            # calculate the score for this boundary
            x, y = sig, template
            score[x0] = np.mean((x-np.mean(x))*(y-np.mean(y)) / \
                                (np.std(x)*np.std(y)))
            
        score = np.pad(score, (pad_st, pad_en))
        
        return score
    #
    # end of get_boundary_scores

    def get_score_matrix(self, feat, ds, st, en):

        # y extrema from the vector to use as cutoff points
        st = np.clip(int(round(st / ds)), 0, None)
        en = np.clip(int(round(en / ds)), None, feat.shape[0]-1)

        # calculate the feat values at each extrema
        # TODO: maybe shouldn't average across entire row? maybe a 3x3 pixel box or something?
        v0 = np.mean(feat[st,:])
        v1 = np.mean(feat[en,:])

        pad_st = st
        pad_en = feat.shape[0] - en

        # loop through each column where we must make a boundary decision
        score = np.zeros(feat.shape)
        for ind in tqdm(range(feat.shape[1])):

            # calculate the score for each boundary decision in this column
            sig = feat[st:en, ind]
            score[:,ind] = self.get_boundary_scores(sig, v0, v1, pad_st, pad_en)
            
        return score

    # finds the feature in the feats matrix with the highest disparity, then uses it to generate the score matrix
    def get_score_matrix_on_feat(self, feats, ds, orig_st, orig_en):
        
        # y extrema from the vector to use as cutoff points
        st = np.clip(int(round(orig_st / ds)), 0, None)
        en = np.clip(int(round(orig_en / ds)), None, feats.shape[1]-1)

        # calculate the feat values at each extrema
        # TODO: maybe shouldn't average across entire row? maybe a 3x3 pixel box or something?
        v0 = np.mean(feats[:, st, :], axis=-1)
        v1 = np.mean(feats[:, en, :], axis=-1)

        # find the feat with the largest disparity
        feat_ind = np.argmax(np.abs((v0-v1)/v0))
        feats = feats[feat_ind]

        score = self.get_score_matrix(feats, ds, orig_st, orig_en)
        
        return score, feat_ind
    #
    # end of get_score_matrix_on_feat
#
# end of HDABProcessor

#
# end of file
