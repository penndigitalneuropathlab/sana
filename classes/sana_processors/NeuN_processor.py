
# system modules
import os

# installed modules
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# custom modules
import sana_io
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor
from sana_geo import Point
from sana_heatmap import Heatmap
from sana_filters import minmax_filter

# debugging modules
from matplotlib import pyplot as plt

class NeuNProcessor(HDABProcessor):
    def __init__(self, fname, frame, debug=False):
        super(NeuNProcessor, self).__init__(fname, frame)
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

        # run the neuron detection algorithms
        # TODO: rename this function, should be a get_features() that calls detect_nurons
        self.detect_neurons(odir, params, disk_r=11, sigma=3,
                            n_iterations=2, close_r=9, open_r=5, soma_r=15, debug=False)
        
        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # save the DAB and HEM images
        self.save_frame(odir, self.dab, 'DAB')
        self.save_frame(odir, self.hem, 'HEM')
        
        # save the params IO to a file
        self.save_params(odir, params)
    #
    # end of run

    def detect_neurons(self, odir, params,
                       disk_r, sigma, n_iterations, close_r, open_r, soma_r, debug=False):
        
        # get the rescale parameters
        hist = self.dab_norm.histogram()        
        vmi = np.argmax(hist)
        vmx = 255

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

        # perform the min-max filtering to maximize centers of cells
        # NOTE: this finds the so called seeds of the image, in a thresholded image with dendrites
        # TODO: inverse this filter
        img_minmax = minmax_filter(255-seed_img, disk_r, sigma, n_iterations, debug)

        # define the sure foreground, small circles at minimums of filter output
        # NOTE: this is done on the somas only image
        candidates = np.where((img_minmax == -1) & (img != 0))
        r0 = 3
        sure_fg = np.zeros_like(img, dtype=np.uint8)
        for i in range(len(candidates[0])):
            sure_fg = cv2.circle(sure_fg,
                                 (candidates[1][i], candidates[0][i]),
                                 r0, color=255, thickness=-1)
        
        # define the sure background, anything 0 and not close in prox. to data
        # NOTE: this is done on the all nueron parts image
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        sure_bg = cv2.dilate(thresh_img, dil_kern, iterations=3)

        # get the unknown areas of the image, not defined by sure bg or fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # run the watershed algo
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        orig_markers = markers.copy()
        rgb = np.stack((thresh_img,)*3, axis=-1) # should be dab_norm
        markers = cv2.watershed(rgb, markers)
        markers[:,0] = 0
        markers[:,-1] = 0
        markers[0,:] = 0
        markers[-1,:] = 0        
        rgb_markers = np.zeros_like(rgb)
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

        tsize = Point(400, 100, True)
        tstep = Point(50, 50, True)
        heatmap = Heatmap(self.dab_norm, cells, tsize, tstep, debug=True)

        # # TODO: this should be like heatmap.get_density()
        # # TODO: could do this all in one loop, add a list of functions to call
        # funcs = [heatmap.density, heatmap.eccentricity, heatmap.circulatiry]
        # titles = ['density', 'eccentricity', 'perimeter to area ratio']
        # feats = heatmap.run(funcs)
        
        # fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
        # ax = axs[0]
        # ax.imshow(self.dab.img)
        # ax.set_title('orig')
        # ax = axs[1]
        # ax.imshow(seed_img, vmin=vmi, vmax=vmx)
        # ax.set_title('soma preprocess')
        # ax = axs[2]        
        # # ax.imshow(img_minmax)
        # # ax.plot(candidates[1], candidates[0], '+', color='red', markersize=3)
        # # ax.set_title('min-max filter (somas)')
        # ax.imshow(orig_markers, vmax=2)
        # ax = axs[3]
        # ax.imshow(rgb_markers)
        # ax.set_title('segmented cells')
        # ax = axs[4]
        # ax.imshow(self.frame.img)
        # ax.plot(candidates[1], candidates[0], '+', color='green', markersize=4)
        # ax.set_title('candidate locs')

        # for i in range(len(feats)):
        #     fig, ax = plt.subplots(1, 1)            
        #     ax.imshow(feats[i], cmap='coolwarm')
        #     ax.set_title(titles[i])

        # plt.show()
            
        annos = []
        for c in cells:
            annos.append(c.to_annotation(self.fname, 'Neuron'))
            
        odir = sana_io.create_odir(odir, 'neurons')
        anno_f = os.path.join(odir, os.path.basename(self.fname).replace('.svs', '_NEURONS.json'))
        sana_io.write_annotations(anno_f, annos)
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
# end of parvalbuminProcessor

#
# end of file
