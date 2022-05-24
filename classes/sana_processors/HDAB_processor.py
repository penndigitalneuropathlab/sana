
# installed modules
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import convolve1d

# custom modules
import sana_io
from sana_frame import Frame, mean_normalize, overlay_thresh, create_mask
from sana_color_deconvolution import StainSeparator
from sana_thresholds import max_dev, kittler
from sana_antibody_processor import Processor
from sana_filters import minmax_filter
from sana_heatmap import Heatmap

# debugging modules
from matplotlib import pyplot as plt
from sana_geo import plot_poly, Point

# generic Processor for H-DAB stained slides
# performs stain separation and rescales the data to 8 bit pixels
class HDABProcessor(Processor):
    def __init__(self, fname, frame, debug):
        super(HDABProcessor, self).__init__(fname, frame)

        # prepare the stain separator
        self.ss = StainSeparator('H-DAB')

        self.debug = debug
        
        # ds = 20
        # img = self.frame.img
        # img = img[::ds, ::ds].astype(float) / 255
        # odimg = np.clip(-np.log10(img), 0, None)
        # odimg[odimg==np.inf] = 0
        # r, g, b = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
        # odr, odg, odb = odimg[:,:,0].flatten(), odimg[:,:,1].flatten(), odimg[:,:,2].flatten()
        # colors = [(r[i], g[i], b[i]) for i in range(len(r))]
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='3d'))
        # ax.scatter(odr, odg, odb, color=colors)
        # x, y, z = self.ss.stain_vector.orig_v[0]
        # ax.quiver(0, 0, 0, x, y, z, color=(x, y, z))
        # x, y, z = self.ss.stain_vector.orig_v[1]
        # ax.quiver(0, 0, 0, x, y, z, color=(x, y, z))
        # plt.show()
        # exit()
        
        # separate out the HEM and DAB stains
        self.stains = self.ss.run(self.frame.img)
        self.hem = Frame(self.stains[:,:,0], frame.lvl, frame.converter)
        self.dab = Frame(self.stains[:,:,1], frame.lvl, frame.converter)

        # rescale the OD stains to 8 bit pixel values
        # NOTE: this uses the physical min/max of the stains based
        #       on the stain vector used
        self.hem.rescale(self.ss.min_od[0], self.ss.max_od[0])
        self.dab.rescale(self.ss.min_od[1], self.ss.max_od[1])

        self.hem_gray = self.hem.copy()
        self.hem_gray.to_gray()
    #
    # end of constructor

    # performs a simple threshold using a manually selected cut off point
    # then runs the %AO process
    def run_manual_ao(self, odir, params):

        # apply the thresholding
        self.manual_dab_thresh = self.dab.copy()
        self.manual_dab_thresh.threshold(self.manual_dab_threshold, 0, 255)

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
        self.save_frame(odir, self.manual_dab_thresh, 'THRESH')
        self.save_frame(odir, self.manual_overlay, 'QC')

        # save the %AO depth curve
        self.save_curve(odir, results['ao_depth'], 'AO')
    #
    # end of run_manual_ao

    # performs normalization, smoothing, and histogram
    # TODO: rename scale to something better
    # TODO: add switches to turn off/on mean_norm, anisodiff, morph
    def run_auto_ao(self, odir, params, scale=1.0, mx=255, debug=False):
        
        # normalize the image
        self.dab_norm = mean_normalize(self.dab)

        # smooth the image
        self.dab_norm.anisodiff()

        # get the histograms
        self.dab_hist = self.dab.histogram()
        self.dab_norm_hist = self.dab_norm.histogram()

        # get the stain threshold
        self.auto_dab_threshold = max_dev(
            self.dab_hist, scale=scale, mx=mx)
        self.auto_dab_norm_threshold = max_dev(
            self.dab_norm_hist, scale=scale, mx=mx, debug=debug)

        # apply the thresholding
        self.auto_dab_norm_thresh = self.dab_norm.copy()
        self.auto_dab_norm_thresh.threshold(self.auto_dab_norm_threshold, 0, 255)
        
        # run the AO process
        results = self.run_ao(self.auto_dab_norm_thresh)

        # store the results of the algorithm
        params.data['area'] = results['area']
        params.data['sub_areas'] = results['sub_areas']        
        params.data['auto_ao'] = results['ao']
        params.data['auto_sub_aos'] = results['sub_aos']
        params.data['auto_stain_threshold'] = self.auto_dab_threshold

        # create the output directory
        odir = sana_io.create_odir(odir, 'auto_ao')

        # save the images used in processing
        self.auto_overlay = overlay_thresh(
            self.frame, self.auto_dab_norm_thresh)        
        self.save_frame(odir, self.dab_norm, 'PROB')
        self.save_frame(odir, self.auto_dab_norm_thresh, 'THRESH')
        self.save_frame(odir, self.auto_overlay, 'QC')
    #
    # end of run_auto_ao

    # TODO: comment and describe parameters, why current values are selected!
    def run_segment(self, odir, params, landmarks,
                disk_r=7, sigma=3,
                close_r=9, open_r=5, debug=False):

        # detect the the hematoxylin cells
        n_iterations = 2
        hem_cells = self.detect_hem_cells(
            params, disk_r, n_iterations, sigma, close_r, open_r, debug)
        
        # calculate the relative density and size of cells throughout the tissue
        tsize = Point(400, 100, True)
        tstep = Point(30, 30, True)
        
        heatmap = Heatmap(self.hem, hem_cells, tsize, tstep, debug=True)
        results = heatmap.run([heatmap.density, heatmap.area])
        cell_dens, cell_size = results[..., 0], results[..., 1]

        # calculate the grayscale intensity throughout the image
        # TODO: should do this slightly differently
        gray = cv2.GaussianBlur(self.hem_gray.img, ksize=(0,0), sigmaX=10, sigmaY=1)
        
        # make sure the landmarks fit in the image boundaries
        landmarks = np.clip(landmarks.astype(int), 0, gray.shape[0]-1)
        
        # detect the CSF to GM boundary with the grayscale intensity
        csf_gm = self.fit_boundary(gray, landmarks[0,0], landmarks[1,1])

        # scale the landmarks to the tiled image
        landmarks = (landmarks/heatmap.tiler.ds).astype(int)
        landmarks = np.clip(landmarks, 0, cell_feat.shape[0]-1)

        # get the L34 and L45 boundaries
        l34, l45 = fit_boundaries(cell_dens, landmarks[2,1], landmarks[3,1])

        # get the L12 boundary
        # NOTE: the l23 boundary may not always be accurate
        #          (i.e. landmark might be placed in L2),
        #       however using this function instead of fit_boundary()
        #       should be more consistent for L12
        l12, l23 = fit_boundaries(cell_dens, landmarks[1,1], landmarks[2,1])

        # get the GM/WM boundary
        # NOTE: same reason we use get_ls here as L12
        l56, gm_wm = fit_boundaries(cell_dens, landmarks[3,1], landmarks[4,1])

        # smooth the boundaries with a moving average filter
        # TODO: this is not really how to do it, theres too much large spikes
        #       that the script attaches to. really want to do waht paul said
        #       the general density of the layer shouldn't change that much 
        Mfilt, Nfilt = 151, 21
        csf_gm_y = convolve1d(csf_gm, np.ones(Mfilt)/Mfilt, mode='reflect')
        boundaries = [convolve1d(x, np.ones(Nfilt)/Nfilt, mode='reflect') \
                  for x in [l12, l23, l34, l45, l56, gm_wm]]

        # scale back to the original resolution
        boundaries_x = np.arange(cell_dens.shape[1]) * heatmap.tiler.ds[0]
        boundaries_y = [x * heatmap.tiler.ds[1] for x in boundaries]
        csf_gm_x = np.linspace(0, boundaries_x[-1], gray.shape[1])

        # generate and store the annotations
        CLASSES = [
            'CSF_BOUNDARY',
            'L1_L2_BOUNDARY','L2_L3_BOOUNDARY',
            'L3_L4_BOUNDARY','L4_L5_BOUNDARY',
            'L5_L6_BOUNDARY','GM_WM_BOUNDARY'
        ]
        csf_gm = Polygon(csf_gm_x, csf_gm_y, False, self.frame.lvl) \
            .to_annotation(ofname, CLASSES[0])
        boundaries = [csf_gm] + [Polygon(boundaries_x, boundaries_y, False, LVL) \
                                 .to_annotation(ofname, CLASSES[i+1]) \
                                 for i in range(len(boundaries_y))]

        # inverse transform the annotations
        [x.transform_inv_poly(
            x, params['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']) \
         for x in boundaries]
        
        # finally, write the predictions
        print(ofname)
        sana_io.write_annotations(ofname, boundaries)
    #
    # end of run_segment

    # this fits a step function to each column in the img
    # TODO: the values of step function are set by the max of the row, is this good?
    def fit_boundary(self, img, st, en):

        # calculate the boundary at each column
        boundary = np.zeros(img.shape[1])
        for j in range(img.shape[1]):

            # extract the signal column
            sig = img[st:en, j]

            if len(sig) == 0:
                boundary[j] = st
                continue
        
            # get the 2 extreme values for the step function
            v0 = np.mean(img[st,:])
            v1 = np.mean(img[en,:])

            # calculate the distance score for each possible index in the signal
            score = np.zeros_like(sig)
            for i in range(sig.shape[0]):

                # create the template signal to compare to the original signal
                template = np.zeros(n)
                template[:x] = v0
                template[x:] = v1

                # calculate the distance between the signals
                score[i] = np.sum(np.abs(sig - template))
            #
            # end of score calculation

            # the boundary val at this column is the min distance index
            boundary[j] = np.argmin(score) + st
        #
        # end of boundary detection

        return boundary
    #
    # end of fit_boundary

    # this fits a sloped pulse function to each column in the img
    # TODO: the values of pulse function are set by the max of the row, is this good?
    def fit_boundaries(self, img, st, en):

        # calculate the width the pulse that is flat
        # TODO: make this a parameter!
        flat_percent = 0.6
        incline_percent = (1 - flat_percent) / 2
        x0p = int((y-x)*incline_percent) + x
        x1p = y - int((y-x)*incline_percent)
        
        # calculate the boundaries at each column
        boundary0, boundary1 = np.zeros(img.shape[1]), np.zeros(img.shape[1])
        for j in range(img.shape[1]):
            
            # extract the signal column
            sig = img[st:en, j]

            # get the 3 extreme values for the pulse function
            v0 = np.mean(img[st,:])
            v1 = np.max(img[st:en, :])
            v2 = np.mean(img[en,:])

            # calculate the distance score for each possible combination of indices
            score = np.full((sig.shape[0], sig.shape[0]), np.inf)
            for x0 in range(sig.shape[0]):
                for x1 in range(sig.shape[0]):
                    
                    # create the template signal to compare to the original signal
                    template = np.zeros(n)
                    template[:x0] = v0
                    template[x0:x0p] = np.linspace(v0, v1, x0p-x)
                    template[x0p:x1p] = v1
                    template[x1p:x1] = np.linspace(v1, v2, y-y0p)
                    template[x1:] = v2
                    
                    # calculate the distance between the signals
                    score[x0][x1] = np.sum(np.abs(sig - template))
            #
            # end of score calculation

            # the boundary vals at this column are the min distance indices
            boundary[j] = np.unravel_index(np.argmin(score), score.shape) + st
        #
        # end of boundary detection

        return boundary
    #
    # end of fit_boundaries
    
    def detect_hem_cells(self, params,
                         disk_r, sigma, n_iterations, close_r, open_r, debug):

        # get the rescale parameters
        hist = self.hem.histogram()        
        vmi = np.argmax(hist)
        vmx = 92

        # # rescale the image
        # self.hem.img = (self.hem.img.astype(float) - mi) / (mx-mi)
        # self.hem.img = np.clip(self.hem.img, 0, None)
        # self.hem.img = (255 * self.hem.img).astype(np.uint8)
        
        # smooth the image
        self.hem.anisodiff()

        self.hem_hist = self.hem.histogram()
        self.hem_threshold = max_dev(self.hem_hist, mx=vmx)
        
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
        img_minmax = minmax_filter(255-img, disk_r, sigma, n_iterations, debug)

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
        markers[:,0] = 0
        markers[:,-1] = 0
        markers[0,:] = 0
        markers[-1,:] = 0        
        rgb_markers = np.zeros_like(rgb_hem)
        colors = [(np.random.randint(10, 255),
                   np.random.randint(10, 255),
                   np.random.randint(10, 255)) \
                  for _ in range(1,np.max(markers))]
        colors = [(0,0,0),(0,0,0)] + colors + [(80,80,80)]
        for j in tqdm(range(rgb_markers.shape[0])):
            for i in range(rgb_markers.shape[1]):
                rgb_markers[j,i] = colors[markers[j,i]]
                
        mframe = self.frame.copy()
        mframe.img = np.where(markers <= 1, 0, 1).astype(np.uint8)
        mframe.get_contours()
        mframe.filter_contours()

        cells = [c.polygon for c in mframe.get_body_contours()]

        if self.debug:        
            gray = self.frame.copy()
            gray.to_gray()

            cell_eccs = np.array([cell.eccentricity() for cell in cells])
            cell_ints = np.array([np.mean(gray.get_tile(*cell.bounding_box())) \
                                  for cell in cells])
            cell_dabs = np.array([np.mean(self.dab.get_tile(*cell.bounding_box())) \
                                  for cell in cells])
        
            fig, axs = plt.subplots(1,3)
            axs[0].hist(cell_eccs, bins=100, range=(0,1))
            axs[0].set_xlabel('eccentricity')
            axs[1].hist(cell_ints, bins=255, range=(0,255))
            axs[1].set_xlabel('grayscale intensity')
            axs[2].hist(cell_dabs, bins=255, range=(0,255))
            axs[2].set_xlabel('dab intensity')
            fig.suptitle('histograms for detected cells')
        
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
        #
        # end of debugging

        return cells
    #
    # end of detect_hem_cells
#
# end of HDABProcessor

#
# end of file
