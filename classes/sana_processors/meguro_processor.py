 
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

    def run(self, odir, params, **kwargs):

        # pre-selected threshold value selected by Dan using
        # multiple images in QuPath
        # NOTE: original value was DAB_OD = 0.3 in QuPath, this
        #       value is calculated from that
        self.manual_dab_threshold = 94
        
        kwargs['scale'] = 0.5
        kwargs['mx'] = 255
        kwargs['open_r'] = 0
        kwargs['close_r'] = 0
        kwargs['min_background'] = 0
        super().run(odir, params, **kwargs)
        
        # # TODO: plot the curves!
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
