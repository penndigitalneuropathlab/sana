
# installed modules
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.spatial import ConvexHull

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, overlay_thresh, create_mask
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_antibody_processor import Processor
from sana_filters import minmax_filter
from sana_heatmap import Heatmap
from sana_geo import Point, Polygon, transform_inv_poly, hull_to_poly

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

        # rescale the OD stains to 8 bit pixel values
        # NOTE: this uses the physical min/max of the stains based
        #       on the stain vector used
        # TODO: this compresses teh digital space, need to scale it back to 0 and 255!
        self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        # calculate the manual dab threshold if a qupath threshold was given
        if self.qupath_threshold:
            self.manual_dab_threshold = \
                (self.qupath_threshold - self.ss.min_od[1]) / \
                (self.ss.max_od[1] - self.ss.min_od[1])

        self.gray = self.frame.copy()
        self.gray.to_gray()
        self.hem_gray = self.hem.copy()
        self.hem_gray.to_gray()
        self.dab_gray = self.dab.copy()
        self.dab_gray.to_gray()
    #
    # end of constructor

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

        # create the output directory
        odir = sana_io.create_odir(odir, 'manual_ao')

        # save the images used in processing
        self.manual_overlay = overlay_thresh(
            self.frame, self.manual_dab_thresh)
        self.save_frame(odir, self.manual_dab_thresh, 'MANUAL_THRESH')
        self.save_frame(odir, self.manual_overlay, 'MANUAL_QC')

        # save the feature signals
        signals = results['signals']
        self.save_signals(odir, signals['normal'], 'MANUAL_NORMAL')
        self.save_signals(odir, signals['main_deform'], 'MANUAL_MAIN_DEFORM')
        if 'sub_deform' in signals:
            self.save_signals(odir, signals['sub_deform'], 'MANUAL_SUB_DEFORM')
    #
    # end of run_manual_ao

    # TODO: rename scale/max
    # function takes in a DAB Frame object, extracting a DAB thresholded image
    def process_dab(self, frame, run_normalize=False, scale=1.0, mx=255, close_r=0, open_r=0, mask = None, debug=False):
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

        if run_normalize:
            # normalize the image
            # TODO: rename mean_normalize --> bckgrnd subtraction (denoising)
            dab = mean_normalize(dab)

            # plot #2
            if debug:
                axs[1].imshow(dab.img)
                axs[1].set_title('Normalized DAB Img')

            # TODO: run anisodiff
            # smooth the image
            dab.anisodiff()

            # plot #3
            if debug:
                axs[2].imshow(dab.img)
                axs[2].set_title('Normalized/Smoothed DAB Img')


        # get the histograms
        dab_hist = dab.histogram()

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
    def run_auto_ao(self, odir, params, scale=1.0, mx=255, open_r=0, close_r=0,):
        # Old DAB Processing code
        # # normalize the image
        self.dab_norm = mean_normalize(self.dab)

        self.auto_dab_thresh_img, self.auto_dab_threshold = \
            self.process_dab(
                self.dab,
                run_normalize = True,
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

        # save the images used in processing
        self.auto_overlay = overlay_thresh(
            self.frame, self.auto_dab_thresh_img)
        #self.save_frame(odir, self.dab_norm, 'AUTO_PROB')
        self.save_frame(odir, self.auto_dab_thresh_img, 'AUTO_THRESH')
        #self.save_frame(odir, self.auto_overlay, 'AUTO_QC')

        # save the feature signals
        signals = results['signals']
        self.save_signals(odir, signals['normal'], 'AUTO_NORMAL')
        self.save_signals(odir, signals['main_deform'], 'AUTO_MAIN_DEFORM')
        if 'sub_deform' in signals:
            self.save_signals(odir, signals['sub_deform'], 'AUTO_SUB_DEFORM')
    #
    # end of run_auto_ao

    # TODO: comment and describe parameters, why current values are selected!
    def run_segment(self, odir, params, landmarks, padding=0,
                disk_r=7, sigma=3,
                close_r=9, open_r=9, debug=False):

        # detect the the hematoxylin cells
        n_iterations = 2
        hem_cells = self.detect_hem_cells(
            params, disk_r, n_iterations, sigma, close_r, open_r, debug=False)

        # write the cell segmentations
        ofname = sana_io.create_filepath(
            self.fname, ext='.json', suffix='CELLS', fpath=odir)
        hem_annos = [x.to_annotation(ofname, 'HEMCELL') for x in hem_cells]
        [transform_inv_poly(
            x, params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) for x in hem_annos]
        sana_io.write_annotations(ofname, hem_annos)

        # tile size and step for the convolution operation to calculate feats
        tsize = Point(300, 200, True)
        tstep = Point(15, 15, True)

        # calculate features from the cell detections
        heatmap = Heatmap(self.hem, hem_cells, tsize, tstep, min_area=0, debug=True)
        feats = heatmap.run([heatmap.density, heatmap.intensity])
        feats_labels = ['DENSITY', 'INTENSITY']

        # calculate the %AO of the cells
        cell_mask = create_mask(
            hem_cells, self.frame.size(), self.frame.lvl, self.frame.converter)
        hm = Heatmap(cell_mask, hem_cells, tsize, tstep)
        ao = hm.ao
        feats = np.concatenate([feats, [ao]], axis=0)
        feats_labels.append('AO')

        feats[feats==np.nan] = 0
        mu = np.mean(feats, axis=(1,2))[:,None,None]
        sigma = np.std(feats, axis=(1,2))[:,None,None]
        sigma[sigma==0] = np.inf
        feats = (feats - mu) / sigma

        # calculate the grayscale intensity throughout the image
        # TODO: should do this slightly differently
        gray = cv2.GaussianBlur(self.hem_gray.img, ksize=(0,0), sigmaX=10, sigmaY=1)
        gray = gray[None, :, :]

        # make sure the landmarks fit in the image boundaries
        landmarks = np.clip(landmarks.astype(int), 0, gray.shape[1]-1)

        # detect the CSF to GM boundary with the grayscale intensity
        csf_gm = self.fit_boundary(gray, landmarks[0,0], landmarks[1,1])

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

        # if self.debug:
        #     fig, axs = plt.subplots(1, feats.shape[0])
        #     if feats.shape[0] == 1:
        #         axs = [axs]
        #     for i in range(feats.shape[0]):
        #         axs[i].imshow(feats[i])
        #         axs[i].set_title(feats_labels[i])
        #         axs[i].plot(l12, color='red')
        #         axs[i].plot(l23, color='orange')
        #         axs[i].plot(l34, color='yellow')
        #         axs[i].plot(l45, color='green')
        #         axs[i].plot(l56, color='blue')
        #         axs[i].plot(gm_wm, color='purple')
        #     plt.show()
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

    # this fits a step function to each column in the img
    # TODO: the values of step function are set by the max of the row, is this good?
    def fit_boundary(self, feats, st, en, v0=None, v1=None, debug=False):

        # calculate the boundary at each column
        boundary = np.zeros(feats.shape[2])
        for j in range(feats.shape[2]):

            # extract the signal column
            sig = feats[:, st:en, j]
            n = sig.shape[1]

            if n == 0:
                boundary[j] = st
                continue

            # get the 2 extreme values for the step function
            if v0 is None:
                v0 = np.mean(feats[:, st,:], axis=1)[:, None]
            if v1 is None:
                v1 = np.mean(feats[:, en,:], axis=1)[:, None]

            # calculate the distance score for each possible index in the signal
            score = np.zeros(sig.shape[1], dtype=float)
            for x0 in range(1, sig.shape[1]-1):

                # create the template signal to compare to the original signal
                template = np.zeros_like(sig)
                template[:, :x0] = v0
                template[:, x0:] = v1

                # calculate the distance between the signals
                corr = []
                for i in range(feats.shape[0]):
                    x, y = sig[i], template[i]
                    corr.append(np.mean((x-np.mean(x))*(y-np.mean(y))) / (np.std(x)*np.std(y)))
                if len(corr) == 1:
                    score[x0] = corr[0]
                else:
                    score[x0] = np.mean(corr)

                if debug:
                    print(corr, flush=True)
                    plt.plot(sig[0])
                    plt.plot(template[0])
                    plt.show()
            #
            # end of score calculation

            # the boundary val at this column is the min distance index
            boundary[j] = np.argmax(score) + st
        #
        # end of boundary detection

        return boundary
    #
    # end of fit_boundary

    # this finds the "best" feature to segment on by looking at the difference
    #  between the values at each landmark
    def fit_boundary_on_feat(self, feats, st, en):

        # get the 2 extreme values for the step function
        v0 = np.mean(feats[:, st,:], axis=1)[:, None]
        v1 = np.mean(feats[:, en,:], axis=1)[:, None]

        # find the feat with the largest disparity
        feat_ind = np.argmax(np.abs((v0-v1)/v0))

        feats = feats[feat_ind][None, ...]
        boundary = self.fit_boundary(feats, st, en)
        return boundary, feat_ind

    # this fits a sloped pulse function to each column in the img
    # TODO: the values of pulse function are set by the max of the row, is this good?
    def fit_boundaries(self, feats, st, en):

        # get the 3 extreme values for the pulse function
        v0 = np.mean(feats[:, st, :], axis=1)[:,None]
        v1 = np.max(np.max(feats[:, st:en, :], axis=1), axis=1)[:,None]
        v2 = np.mean(feats[:, en, :], axis=1)[:,None]

        # find the feat with the largest disparity
        # feat_ind = np.argmax(np.abs((v0-v1)/v0) + np.abs((v1-v2)/v2))
        # feats = feats[feat_ind][None, ...]
        # v0 = np.mean(feats[:, st, :], axis=1)[:,None]
        # v1 = np.max(np.max(feats[:, st:en, :], axis=1), axis=1)[:,None]
        # v2 = np.mean(feats[:, en, :], axis=1)[:,None]

        # calculate the width the pulse that is flat
        # TODO: make this a parameter!
        flat_percent = 0.6
        incline_percent = (1 - flat_percent) / 2

        # calculate the boundaries at each column
        boundary0, boundary1 = np.zeros(feats.shape[2]), np.zeros(feats.shape[2])
        for j in range(feats.shape[2]):

            # extract the signal column
            sig = feats[:, st:en, j]
            n = sig.shape[1]

            # calculate the distance score for each possible combination of indices
            score = np.zeros((sig.shape[1], sig.shape[1]), dtype=float)
            for x0 in range(1, sig.shape[1]):
                for x1 in range(x0+4, sig.shape[1]):

                    x0p = int((x1-x0)*incline_percent) + x0
                    x1p = x1 - int((x1-x0)*incline_percent)

                    v01 = np.linspace(v0[:,0], v1[:,0], x0p-x0).T
                    v12 = np.linspace(v1[:,0], v2[:,0], x1-x1p).T

                    # create the template signal to compare to the original signal
                    template = np.zeros_like(sig)
                    template[:, :x0] = v0
                    template[:, x0:x0p] = v01
                    template[:, x0p:x1p] = v1
                    template[:, x1p:x1] = v12
                    template[:, x1:] = v2

                    # calculate the distance between the signals
                    corr = []
                    for i in range(feats.shape[0]):
                        x, y = sig[i], template[i]
                        corr.append(np.mean((x-np.mean(x))*(y-np.mean(y))) / (np.std(x)*np.std(y)))
                    score[x0][x1] = np.mean(corr)
            #
            # end of score calculation

            # the boundary vals at this column are the min distance indices
            inds = np.unravel_index(np.argmax(score), score.shape)
            boundary0[j] = inds[0] + st
            boundary1[j] = inds[1] + st
        #
        # end of boundary detection

        return boundary0, boundary1
    #
    # end of fit_boundaries

    def detect_hem_cells(self, params,
                         disk_r, sigma, n_iterations, close_r, open_r, debug=False):

        # smooth the image
        # self.hem.anisodiff()

        self.hem_hist = self.hem.histogram()
        self.hem_threshold = max_dev(self.hem_hist, mx=110)

        # get the image array
        img = self.hem.img.copy()[:,:,0]

        # do some thresholding and morph filters to
        # 1) remove faint objects (ambiguous)
        # 2) delete tiny objects (too small/fragments)
        # 3) close holes in center of faint neurons
        thresh_img = img.copy()
        thresh_img = np.where(thresh_img < self.hem_threshold, 0, 255)
        thresh_img = thresh_img.astype(np.uint8)
        close_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_r, close_r))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, close_kern)
        open_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_r, open_r))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, open_kern)

        # mask out all unwanted data as described above
        img[thresh_img == 0] = 0

        # define the sure background, anything 0 and not close in prox. to data
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        sure_bg = cv2.dilate(thresh_img, dil_kern, iterations=3)

        # perform the min-max filtering to maximize centers of cells
        # TODO: inverse this filter
        img_minmax = minmax_filter(255-img, disk_r, sigma, n_iterations)

        # define the sure foreground, small circles at minimums of filter output
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
        rgb_hem = np.stack((self.hem.img[:,:,0],)*3, axis=-1)
        markers = cv2.watershed(rgb_hem, markers)
        markers[markers==1] = 0
        markers[:,0] = 0
        markers[:,-1] = 0
        markers[0,:] = 0
        markers[-1,:] = 0

        # cells = []
        # for i in tqdm(range(2, np.max(markers))):
        #     points = np.where(markers == i)
        #     points = np.vstack([points[1], points[0]]).T
        #     ch = ConvexHull(points)
        #     xy = points[ch.vertices]
        #     cells.append(Polygon(xy[:,0], xy[:,1], False, 0))
        #     #cells.append(hull_to_poly(ch, points))

        # TODO: this method doesn't separate some touching cells, but its fast
        mframe = self.frame.copy()
        mframe.img = np.where(markers <= 1, 0, 1).astype(np.uint8)
        mframe.get_contours()
        mframe.filter_contours()

        cells = [c.polygon for c in mframe.get_body_contours()]

        if debug:

            rgb_markers = np.zeros_like(rgb_hem)
            colors = [(np.random.randint(10, 255),
                       np.random.randint(10, 255),
                       np.random.randint(10, 255)) \
                      for _ in range(1,np.max(markers))]
            colors = [(0,0,0)] + colors + [(80,80,80)]
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

        if debug:
            gray = self.frame.copy()
            gray.to_gray()

            cell_eccs = np.array([cell.eccentricity() for cell in cells])
            cell_ints = np.array([np.mean(gray.get_tile(*cell.bounding_box())) \
                                  for cell in cells])
            cell_dabs = np.array([np.mean(self.dab.get_tile(*cell.bounding_box())) \
                                  for cell in cells])
            cell_sizes = np.array([cell.area() for cell in cells])


            fig, axs = plt.subplots(1,4)
            axs[0].hist(cell_eccs, bins=100, range=(0,1))
            axs[0].set_xlabel('eccentricity')
            axs[1].hist(cell_ints, bins=255, range=(0,255))
            axs[1].set_xlabel('grayscale intensity')
            axs[2].hist(cell_dabs, bins=255, range=(0,255))
            axs[2].set_xlabel('dab intensity')
            axs[3].hist(cell_sizes, bins=100)
            axs[3].set_xlabel('area')
            fig.suptitle('histograms for detected cells')
            plt.show()
        #
        # end of debugging

        return cells
    #
    # end of detect_hem_cells
#
# end of HDABProcessor

#
# end of file
