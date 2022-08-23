
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
from sana_geo import Point, transform_inv_poly, Polygon
from sana_heatmap import Heatmap
from sana_filters import minmax_filter
from sana_frame import mean_normalize, create_mask, overlay_thresh

# debugging modules
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sana_geo import plot_poly

class NeuNProcessor(HDABProcessor):
    def __init__(self, fname, frame, roi_type, Nsamp, debug=False):
        super(NeuNProcessor, self).__init__(fname, frame, roi_type=roi_type, Nsamp=Nsamp, debug=debug)
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
        self.run_auto_ao(odir, params, scale=1.0, mx=90)

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

    def run_neurons(self, odir, params, main_roi, disk_r=7, sigma=3,
                    close_r=9, open_r=9, soma_r=15, debug=False):
        
        # detect the the hematoxylin cells
        n_iterations = 2
        neurons = self.detect_neurons(
            odir, params, disk_r, n_iterations, sigma, close_r, open_r, soma_r, debug=False)

        neurons = [n for n in neurons if n.inside(main_roi)]

        if len(neurons) < 10:
            params.data['grn_ao'] = np.nan
            params.data['grn_sub_aos'] = np.nan
            params.data['pyr_ao'] = np.nan
            params.data['pyr_sub_aos'] = np.nan
            return neurons
        
        # calculate the feats of the data
        # TODO: remove area here and test
        feats = np.zeros((5, len(neurons)), dtype=float)
        feats[0] = [x.area() for x in neurons]
        feats[1] = [x.minor() for x in neurons]
        feats[2] = [x.major() for x in neurons]
        feats[3] = [x.eccentricity() for x in neurons]
        feats[4] = [x.circularity() for x in neurons]  
        feats = feats.T

        # split the neurons into K different clusters
        K = 4
        clf = KMeans(n_clusters=K)
        clf = clf.fit(feats)
        labels = clf.labels_
        clusters = [[] for _ in range(K)]
        for i in range(len(neurons)):
            clusters[labels[i]].append(neurons[i])

        # get the cluster with the lowest eccentricity, most likely granular cluster
        grn_ind = np.argmin(clf.cluster_centers_[:,3])

        # combine the other clusters into the pyramidal cluster
        grn = clusters[grn_ind]
        pyr = []
        for i in range(len(clusters)):
            if i != grn_ind:
                pyr.extend(clusters[i])

        # get the thresholded masks of the granular and pyramidal classes
        grn_mask = create_mask(grn, self.frame.size(), self.frame.lvl, self.frame.converter)
        pyr_mask = create_mask(pyr, self.frame.size(), self.frame.lvl, self.frame.converter)
        tot_mask = create_mask(grn+pyr, self.frame.size(), self.frame.lvl, self.frame.converter)
        
        # run the AO process over grn and pyr
        grn_results = self.run_ao(grn_mask, grn)
        pyr_results = self.run_ao(pyr_mask, pyr)
        tot_results = self.run_ao(tot_mask, grn+pyr)

        # store the results! 
        params.data['grn_ao'] = grn_results['ao']
        params.data['grn_sub_aos'] = grn_results['sub_aos']
        params.data['pyr_ao'] = pyr_results['ao']
        params.data['pyr_sub_aos'] = pyr_results['sub_aos']
        params.data['tot_ao'] = tot_results['ao']
        params.data['tot_sub_aos'] = tot_results['sub_aos']

        # create and store the debugging overlay
        overlay = self.frame.copy()
        cols = ['blue', 'red']    
        overlay = overlay_thresh(overlay, grn_mask, alpha=0.5, color=cols[0])
        overlay = overlay_thresh(overlay, pyr_mask, alpha=0.5, color=cols[1])

        # save the original frame
        self.save_frame(odir, overlay, 'GRNPYR')

        # save the depth curves
        # TODO: want to also store the undeformed verisons?
        # TODO: not really ao_depth, since these are actually density curves, ao curves are calculated in the auto/manual
        self.save_curve(odir, grn_results['ao_depth'], 'GRN')
        self.save_curve(odir, pyr_results['ao_depth'], 'PYR')
        self.save_curve(odir, tot_results['ao_depth'], 'TOT')        
        
        if debug:            
            custom_lines = [Line2D([0],[0], color=col, lw=4) for col in cols]
            fig, ax = plt.subplots(1,1)
            ax.imshow(overlay.img)
            ax.legend(custom_lines, ['Granular', 'Pyramidal'])
            plt.axis('off')
            plt.show()
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
        boundaries = [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in boundaries]

        # finally, write the predictions
        sana_io.write_annotations(ofname, boundaries)

        # save the feature files
        for i in range(feats.shape[0]):
            self.save_array(odir, feats[i], feats_labels[i])
    #
    # end of run_segment
    
    def detect_neurons(self, odir, params,
                       disk_r, n_iterations, sigma, close_r, open_r, soma_r, debug=False):
        
        # get the image array
        img = self.dab_norm.img.copy()[:,:,0]

        # get the thresholded images, one for somas only, and one for all neuron parts
        somas_img = self.get_thresh_img(img, self.auto_dab_norm_threshold, close_r, soma_r)
        thresh_img = self.get_thresh_img(img, self.auto_dab_norm_threshold, close_r, open_r)

        # mask for only somas, this will be used to find the seeds of the image
        seed_img = img.copy()
        seed_img[somas_img == 0] = 0

        # mask for all neuron parts, this will be used for the final segmentation
        neur_img = img.copy()
        neur_img[thresh_img == 0] = 0

        # define the sure background, anything 0 and not close in prox. to data
        # NOTE: this is done on the all nueron parts image
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        sure_bg = cv2.dilate(thresh_img, dil_kern, iterations=3)
        
        # perform the min-max filtering to maximize centers of cells
        # NOTE: this finds the so called seeds of the image, in a thresholded image with dendrites
        # TODO: inverse this filter
        img_minmax = minmax_filter(255-seed_img, disk_r, sigma, n_iterations, self.debug)

        # define the sure foreground, small circles at minimums of filter output
        # NOTE: this is done on the somas only image
        candidates = np.where((img_minmax == -1) & (img != 0))
        r0 = 3
        sure_fg = np.zeros_like(img, dtype=np.uint8)
        for i in range(len(candidates[0])):
            sure_fg = cv2.circle(sure_fg,
                                 (candidates[1][i], candidates[0][i]),
                                 r0, color=255, thickness=-1)
        
        # get the unknown areas of the image, not defined by sure bg or fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # run the watershed algo
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        rgb = np.stack((thresh_img,)*3, axis=-1) # should be dab_norm
        markers = cv2.watershed(rgb, markers)
        markers[:,0] = 0
        markers[:,-1] = 0
        markers[0,:] = 0
        markers[-1,:] = 0

        # TODO: this method doesn't separate some touching cells, but its fast        
        mframe = self.frame.copy()
        mframe.img = np.where(markers <= 1, 0, 1).astype(np.uint8)
        mframe.get_contours()
        mframe.filter_contours()

        neurons = [c.polygon for c in mframe.get_body_contours()]
        
        if debug:
            rgb_markers = np.zeros_like(rgb)
            colors = [(np.random.randint(10, 255),
                       np.random.randint(10, 255),
                       np.random.randint(10, 255)) \
                      for _ in range(1,np.max(markers))]
            colors = [(0,0,0),(0,0,0)] + colors + [(80,80,80)]
            for j in tqdm(range(rgb_markers.shape[0])):
                for i in range(rgb_markers.shape[1]):
                    rgb_markers[j,i] = colors[markers[j,i]]

            fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
            axs[0].imshow(self.frame.img)
            axs[0].set_title('orig')
            axs[1].imshow(img, vmin=vmi, vmax=vmx)
            axs[1].set_title('preprocess')
            axs[2].imshow(img_minmax)
            axs[2].plot(candidates[1], candidates[0], '+', color='red', markersize=3)
            axs[2].set_title('min-max filter')
            axs[3].imshow(rgb_markers)
            axs[3].set_title('segmented cells')
            axs[4].imshow(self.frame.img)
            axs[4].plot(candidates[1], candidates[0], '+', color='green', markersize=4)
            axs[4].set_title('candidate locs')
            plt.show()
                    

        # tsize = Point(400, 100, True)
        # tstep = Point(50, 50, True)
        # heatmap = Heatmap(self.dab_norm, cells, tsize, tstep, debug=True)

        # # TODO: this should be like heatmap.get_density()
        # # TODO: could do this all in one loop, add a list of functions to call
        # funcs = [heatmap.density, heatmap.eccentricity, heatmap.circulatiry]
        # titles = ['density', 'eccentricity', 'perimeter to area ratio']
        # feats = heatmap.run(funcs)

        # write the cell segmentations
        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='NEURONS', fpath=odir)
        neuron_annos = [x.to_annotation(ofname, 'NEURON') for x in neurons]
        neuron_annos = [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in neuron_annos]
        sana_io.write_annotations(ofname, neuron_annos)

        return neurons
    #
    # end of detect_neurons    
        
    # do some thresholding and morph filters to
    # 1) remove faint objects (ambiguous)
    # 2) close holes in center of faint neurons        
    # 3) delete tiny objects (too small/fragments)
    # NOTE: in 3) making the open_r big deletes dendrites
    def get_thresh_img(self, img, thresh, close_r, open_r):
        img = np.where(img < thresh, 0, 255).astype(np.uint8)
        close_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_r, close_r))
        open_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_r, open_r))        
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_kern)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, open_kern)

        return img
    #
    # end of get_thresh_img
#
# end of NeuNProcessor

#
# end of file
