
# system modules
import os

# installed modules
import numpy as np

# custom modules
from sana_thresholds import max_dev, kittler
from sana_processors.HDAB_processor import HDABProcessor

# debugging modules
from matplotlib import pyplot as plt

class MeguroProcessor(HDABProcessor):
    def __init__(self, fname, frame, logger, **kwargs):
        super(MeguroProcessor, self).__init__(fname, frame, logger, **kwargs)
    #
    # end of constructor

    def run(self, odir, roi_odir, first_run, params, main_roi, sub_rois=[]):
        self.generate_masks(main_roi, sub_rois)

        # save the original frame
        if self.save_images:
            self.save_frame(odir, self.frame, 'ORIG')
        
        # generate the auto AO results
        self.run_auto_ao(odir, params, scale=1.0, mx=90)

        # save the original frame
        self.save_frame(odir, self.frame, 'ORIG')

        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94

        # generate the manually curated AO results
        self.run_manual_ao(odir, params)
        
        # save the params IO to a file
        self.save_params(odir, params)

        # TODO: plot the curves!
        # slide_name = os.path.splitext(os.path.basename(self.fname))[0]
        # signal = np.load(os.path.join(odir, slide_name+'_AUTO_MAIN_DEFORM.npy'))[0][0]
        # ao = params.data['auto_ao']
        # depth = np.linspace(0, self.frame.img.shape[0], signal.shape[0])
        
        # fig, axs = plt.subplots(1,2, sharey=True, figsize=(16,9), gridspec_kw={'width_ratios': [3,1]})
        # axs[0].imshow(self.frame.img)
        # axs[1].axvline(100*ao, linestyle='--', color='gray')        
        # axs[1].plot(100*signal, depth, color='red')
        # axs[0].set_ylabel(r'Pixels (1 pixel per 0.5045 $\mu$m)')
        # axs[1].set_xlabel('%Area Occupied')
        # plt.subplots_adjust(left=0.1,
        #                     bottom=0.1,
        #                     right=0.9,
        #                     top=0.9,
        #                     wspace=0.01,
        #                     hspace=0.4)        
        
        # fig, axs = plt.subplots(2,1, sharex=True, figsize=(16,9), gridspec_kw={'height_ratios': [3,1]})
        # axs[0].imshow(self.frame.img.transpose(1,0,2))
        # axs[1].plot(depth, signal)
        # plt.subplots_adjust(left=0.1,
        #                     bottom=0.1,
        #                     right=0.9,
        #                     top=0.9,
        #                     wspace=0.4,
        #                     hspace=0.01)        
        # plt.show()
    #
    # end of run
#
# end of meguroProcessor

#
# end of file
