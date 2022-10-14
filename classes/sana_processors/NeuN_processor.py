
# system modules
import os

# installed modules
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import convolve1d
from sklearn.cluster import KMeans

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point, transform_inv_poly, Polygon, Neuron
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from sana_frame import mean_normalize, create_mask, overlay_thresh

# debugging modules
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sana_geo import plot_poly

class NeuNProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(NeuNProcessor, self).__init__(fname, frame, logger, **kwargs)
        self.debug = debug
    #
    # end of constructor

    def run(self, odir, params, main_roi, sub_rois=[]):

        self.mask_frame(main_roi, sub_rois)
        
        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94
        
        # generate the manually curated AO results
        self.run_manual_ao(odir, params)

        # generate the auto AO results
        # NOTE: this threshold is super lenient, additionally we remove very tiny objects
        self.run_auto_ao(odir, params, scale=1.0, mx=90, open_r=7)

        # detect and analyze the neurons in the ROI
        self.run_neurons(odir, params, main_roi, debug=False)
        
        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    def run_neurons(self, odir, params, main_roi, debug=False):
        
        # instance segment the neurons
        disk_r = 7
        sigma = 3
        n_iterations = 1
        close_r = 9
        open_r = 9
        clean_r = 15
        cells = self.segment_cells(
            self.dab_norm, self.auto_dab_norm_threshold, disk_r, sigma, n_iterations, close_r, open_r, clean_r, debug=False)

        neurons = [Neuron(cell, self.fname, self.main_roi, confidence=1.0) for cell in cells]
        
        # write the neuron segmentations
        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='NEURONS', fpath=odir)
        neuron_annos = [x.polygon.to_annotation(ofname, 'NEURON') for x in neurons]
        [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in neuron_annos]
        sana_io.write_annotations(ofname, neuron_annos)

        # only include neurons inside the main ROI
        neurons = [n for n in neurons if n.polygon.inside(main_roi)]

        # very few neurons detected, just report NaN
        if len(neurons) < 10:
            params.data['grn_ao'] = np.nan
            params.data['grn_sub_aos'] = np.nan
            params.data['pyr_ao'] = np.nan
            params.data['pyr_sub_aos'] = np.nan
            return neurons

        # grab the features of the neurons
        feats = np.concatenate([x.feats for x in neurons], axis=0)
        
        # load the pre-trained model
        # NOTE: this isn't a wildcat model but no better place for it right now
        clf = pickle.load(open(os.path.join(os.environ['SANAHOME'], 'classes', 'wildcat', 'RF_model.dat'), 'rb'))

        # TODO: this is hardcoded bc idk how to save this somewhere convenient, would be nice to wrap the threshold and teh .dat file into some object that a class can load, or just hardcode it in the classifier class
        thresh = 0.62

        # classify neurons as either pyramidal or non
        probs = np.array(clf.predict_proba(feats))[:,1]
        pyr_inds = probs > thresh
        non_inds = probs <= thresh
        pyr_neurons = [neurons[i].polygon for i in pyr_inds]
        non_neurons = [neurons[i].polygon for i in non_inds]
        tot_neurons = [neurons[i].polygon for i in range(len(neurons))]
        
        # get the thresholded masks of the neurons
        pyr_mask = create_mask(pyr_neurons, self.frame.size(), self.frame.lvl, self.frame.converter)
        non_mask = create_mask(non_neurons, self.frame.size(), self.frame.lvl, self.frame.converter)
        tot_mask = create_mask(tot_neurons, self.frame.size(), self.frame.lvl, self.frame.converter)
        tot_outline = create_mask(tot_neurons, self.frame.size(), self.frame.lvl, self.frame.converter, outlines_only=True)
        
        # run the AO process over grn and pyr
        pyr_results = self.run_ao(pyr_mask, pyr_neurons)
        non_results = self.run_ao(non_mask, non_neurons)        
        tot_results = self.run_ao(tot_mask, tot_neurons)

        # store the results!
        params.data['pyr_ao'] = pyr_results['ao']
        params.data['non_ao'] = non_results['ao']
        params.data['tot_ao'] = tot_results['ao']        
        params.data['pyr_sub_aos'] = pyr_results['sub_aos']        
        params.data['non_sub_aos'] = non_results['sub_aos']
        params.data['tot_sub_aos'] = tot_results['sub_aos']

        # create and store the debugging overlay
        overlay = self.frame.copy()
        colors = ['red', 'blue']
        overlay = overlay_thresh(overlay, pyr_mask, alpha=0.5, color=colors[0])
        overlay = overlay_thresh(overlay, non_mask, alpha=0.5, color=colors[1])
        overlay = overlay_thresh(overlay, tot_outline, alpha=1.0, color='black')

        # save the original frame
        self.save_frame(odir, overlay, 'PYR')

        # save the feature signals
        pyr_signals = pyr_results['signals']        
        non_signals = non_results['signals']
        tot_signals = tot_results['signals']
        self.save_signals(odir, pyr_signals['normal'], 'PYR_NORMAL')        
        self.save_signals(odir, non_signals['normal'], 'NON_NORMAL')
        self.save_signals(odir, tot_signals['normal'], 'TOT_NORMAL')
        self.save_signals(odir, pyr_signals['main_deform'], 'PYR_MAIN_DEFORM')
        self.save_signals(odir, non_signals['main_deform'], 'NON_MAIN_DEFORM')
        self.save_signals(odir, tot_signals['main_deform'], 'TOT_MAIN_DEFORM')
        self.save_signals(odir, pyr_signals['sub_deform'], 'PYR_SUB_DEFORM')
        self.save_signals(odir, non_signals['sub_deform'], 'NON_SUB_DEFORM')
        self.save_signals(odir, tot_signals['sub_deform'], 'TOT_SUB_DEFORM')

        if self.logger.plots:
            custom_lines = [Line2D([0],[0], color=color, lw=4) for color in colorss]
            fig, ax = plt.subplots(1,1)
            ax.imshow(overlay.img)
            ax.legend(custom_lines, ['Pyramidal', 'Non-Pyramidal'])
            plt.axis('off')
            plt.show()

            exit()        
    #
    # end of run_neurons
    
    def run_segment(self, odir, params, landmarks, padding=0,
                    disk_r=7, sigma=3,
                    close_r=9, open_r=9, soma_r=15, debug=False):

        # normalize the image
        self.dab_norm = mean_normalize(self.dab)

        # smooth the image
        self.dab_norm.anisodiff()

        # get the histograms
        self.dab_hist = self.dab.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        # get the stain threshold
        scale = 1.0
        mx = 90
        self.auto_dab_threshold = max_dev(
            self.dab_hist, scale=scale, mx=mx)
        self.auto_dab_norm_threshold = max_dev(
            self.dab_norm_hist, scale=scale, mx=mx, debug=debug)
        
        # detect the the hematoxylin cells
        n_iterations = 2
        neurons = self.detect_neurons(
            odir, params, disk_r, n_iterations, sigma, close_r, open_r, soma_r, debug=False)

        # tile size and step for the convolution operation to calculate feats
        tsize = Point(300, 200, True)
        tstep = Point(15, 15, True)

        # calculate features from the cell detections
        heatmap = Heatmap(self.dab_norm, neurons, tsize, tstep, min_area=0, debug=True)
        feats = heatmap.run([heatmap.density, heatmap.intensity])
        feats_labels = ['DENSITY', 'INTENSITY']

        # calculate the %AO of the cells
        neuron_mask = create_mask(
            neurons, self.frame.size(), self.frame.lvl, self.frame.converter)
        hm = Heatmap(neuron_mask, neurons, tsize, tstep)
        ao = hm.ao
        feats = np.concatenate([feats, [ao]], axis=0)
        feats_labels.append('AO')

        # standardize the features
        feats[feats==np.nan] = 0
        mu = np.mean(feats, axis=(1,2))[:,None,None]
        sigma = np.std(feats, axis=(1,2))[:,None,None]
        sigma[sigma==0] = np.inf
        feats = (feats - mu) / sigma

        # calculate the grayscale intensity throughout the image
        # TODO: should do this slightly differently
        gray = cv2.GaussianBlur(self.gray.img, ksize=(0,0), sigmaX=10, sigmaY=1)
        gray = gray[None, :, :]

        # make sure the landmarks fit in the image boundaries
        landmarks = np.clip(landmarks.astype(int), 0, gray.shape[1]-1)

        # detect the CSF to GM boundary with the grayscale intensity
        csf_gm = self.fit_boundary(gray, landmarks[0,1], landmarks[1,1], v0=255, v1=0)

        # scale the landmarks to the tiled image
        landmarks = (landmarks/heatmap.tiler.ds).astype(int)
        landmarks = np.clip(landmarks, 0, feats.shape[1]-1)

        # get the L34 and L45 boundaries
        l34, l45 = self.fit_boundaries(feats, landmarks[2,1], landmarks[3,1])

        # get the L12 boundary
        # NOTE: the l23 boundary may not always be accurate
        #          (i.e. landmark might be placed in L2),
        #       however using this function instead of fit_boundary()
        #       should be more consistent for L12
        l12, l23 = self.fit_boundaries(feats, landmarks[1,1], int(np.mean(l34)))

        # get the GM/WM boundary
        # NOTE: same reason we use get_ls here as L12
        l56, gm_wm = self.fit_boundaries(feats, int(np.mean(l45)), landmarks[4,1])

        # gm_wm, feat_ind = self.fit_boundary_on_feat(feats, landmarks[3,1], landmarks[4,1])
        gm_wm = self.fit_boundary(feats, landmarks[3,1], landmarks[4,1])
        
        # TODO: need to clip the boundaries to the padding

        # smooth the boundaries with a moving average filter
        # TODO: this is not really how to do it, theres too much large spikes
        #       that the script attaches to. really want to do waht paul said
        #       the general density of the layer shouldn't change that much
        Mfilt, Nfilt = 151, 21
        csf_gm_y = convolve1d(csf_gm, np.ones(Mfilt)/Mfilt, mode='reflect')
        boundaries = [convolve1d(x, np.ones(Nfilt)/Nfilt, mode='reflect') \
                  for x in [l12, l23, l34, l45, l56, gm_wm]]

        # scale back to the original resolution
        boundaries_x = np.arange(feats.shape[2]) * heatmap.tiler.ds[0]
        boundaries_y = [x * heatmap.tiler.ds[1] for x in boundaries]
        csf_gm_x = np.linspace(0, boundaries_x[-1], gray.shape[2])

        fig, axs = plt.subplots(1,1)
        axs.imshow(gray[0])
        axs.plot(csf_gm_x, csf_gm_y, color='red')
        
        csf_gm_x /= heatmap.tiler.ds[0]
        csf_gm_y /= heatmap.tiler.ds[1]
        boundaries_x /= heatmap.tiler.ds[0]
        boundaries_y = [x / heatmap.tiler.ds[1] for x in boundaries_y]
        
        # fig, axs = plt.subplots(1, feats.shape[0])
        # if feats.shape[0] == 1:
        #     axs = [axs]
        # for i in range(feats.shape[0]):
        #     axs[i].imshow(feats[i])
        #     axs[i].set_title(feats_labels[i])
        #     axs[i].plot(csf_gm_x, csf_gm_y, color='red')
        #     axs[i].plot(boundaries_x, boundaries_y[0], color='orange')
        #     axs[i].plot(boundaries_x, boundaries_y[1], color='yellow')
        #     axs[i].plot(boundaries_x, boundaries_y[2], color='green')
        #     axs[i].plot(boundaries_x, boundaries_y[3], color='blue')
        #     # axs[i].plot(boundaries_x, boundaries_y[4], color='orange')
        #     axs[i].plot(boundaries_x, boundaries_y[5], color='purple')
        # plt.show()

        csf_gm_x *= heatmap.tiler.ds[0]
        csf_gm_y *= heatmap.tiler.ds[1]
        boundaries_x *= heatmap.tiler.ds[0]
        boundaries_y = [x * heatmap.tiler.ds[1] for x in boundaries]
        
        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='GM', fpath=odir)

        # generate and store the annotations
        CLASSES = [
            'CSF_BOUNDARY',
            'L1_L2_BOUNDARY','L2_L3_BOOUNDARY',
            'L3_L4_BOUNDARY','L4_L5_BOUNDARY',
            'L5_L6_BOUNDARY','GM_WM_BOUNDARY'
        ]
        LVL = 0
        csf_gm = Polygon(csf_gm_x, csf_gm_y, False, self.frame.lvl) \
            .to_annotation(ofname, CLASSES[0])
        boundaries = [csf_gm] + [Polygon(boundaries_x, boundaries_y[i], False, LVL) \
                                 .to_annotation(ofname, CLASSES[i+1]) \
                                 for i in range(len(boundaries_y))]

        # inverse transform the annotations
        [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in boundaries]

        # finally, write the predictions
        sana_io.write_annotations(ofname, boundaries)

        # save the feature files
        for i in range(feats.shape[0]):
            self.save_array(odir, feats[i], feats_labels[i])
    #
    # end of run_segment
#
# end of NeuNProcessor

#
# end of file
