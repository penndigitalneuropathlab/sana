# system modules
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# installed modules
import torch
from torchvision import transforms
import scipy
import numpy as np
from PIL import Image

# custom modules
from wildcat.unet_wildcat import resnet50_wildcat_upsample

# debugging modules
from matplotlib import pyplot as plt

class Model:
    def __init__(self, model_path, frame, num_classes, kmax=0.02, alpha=0.7, num_maps=4, kmin=0.0, debug=None):
        self.model_path = model_path
        self.frame = frame
        self.num_classes = num_classes
        self.debug = debug

        # grab the device to run the model on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load the model
        self.model = resnet50_wildcat_upsample(self.num_classes, pretrained=False,
            kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

        # size of patch to be inputted to the model
        self.patch_raw = 112

        # size of window to load in
        self.window_raw = 1120

        # padding, relative to patch_size to add to the window
        self.padding_rel = 1.0
        self.padding_raw = int(self.padding_rel * self.patch_raw)

        # amount wildcat shrinks input images when mapping to segmentations
        self.wildcat_shrinkage = 2

        # don't want to store large output images
        # TODO: do we need this?
        self.extra_shrinkage = 1

        # size of output pixel (in input pixels)
        self.out_pix_size = self.wildcat_shrinkage * self.extra_shrinkage
        #self.out_pix_size = self.extra_shrinkage

        # output size for each window
        self.window_out = int(self.window_raw / self.out_pix_size)

        # padding size for the output
        self.padding_out = int(self.padding_rel * self.patch_raw / self.out_pix_size)

        # size of image to process
        self.slide_dim = np.array(self.frame.img.shape[:2])

        # total number of non-overlapping windows to process
        self.n_win = np.ceil(self.slide_dim / self.window_raw).astype(int)

        # output image size
        self.out_dim = (self.n_win * self.window_out).astype(int)

        self.extra_scaled = (self.window_raw * self.n_win) / self.frame.img.shape[:-1]
        self.true_out_dim = (self.out_dim / self.extra_scaled).astype(int)
    #
    # end of constructor

    def run(self, debug=False):

        # initialize the output array
        output = np.zeros((self.num_classes, self.out_dim[0], self.out_dim[1]))
        
        # range of pix to scan
        u_range, v_range = (0, self.n_win[0]), (0, self.n_win[1])

        # loop over windows
        for u in range(u_range[0], u_range[1]):
            for v in range(v_range[0], v_range[1]):

                # get coords of the window in raw pixels
                x, y, w = u * self.window_raw, v * self.window_raw, self.window_raw

                # subtract the padding
                xp0, yp0, wp = x - self.padding_raw, y - self.padding_raw, self.window_raw + 2*self.padding_raw
                xp1, yp1 = xp0+wp, yp0+wp
                xpad0, xpad1, ypad0, ypad1 = 0,0,0,0
                if xp0 < 0:
                    xpad0 = 0 - xp0
                    xp0 = 0
                if yp0 < 0:
                    ypad0 = 0 - yp0
                    yp0 = 0
                if xp1 > self.frame.img.shape[0]:
                    xpad1 = xp1 - self.frame.img.shape[0]
                    xp1 = self.frame.img.shape[0]
                if yp1 > self.frame.img.shape[1]:
                    ypad1 = yp1 - self.frame.img.shape[1]
                    yp1 = self.frame.img.shape[1]

                chunk = self.frame.img[xp0:xp1, yp0:yp1]
                chunk = np.pad(chunk, [(xpad0,xpad1),(ypad0,ypad1),(0,0)], mode='constant', constant_values=255)
                chunk = Image.fromarray(chunk)

                # compute the desired size of input
                # TODO: input_size should be here as a ratio
                wwc = int(wp * self.patch_raw / self.patch_raw)

                # resample the chunk for the two networks
                tran = transforms.Compose([
                    transforms.Resize((wwc, wwc)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                # convert the read chunk to tensor format
                with torch.no_grad():

                    # Apply transofrms and turn into correct size torch tensor
                    chunk_tensor = torch.unsqueeze(tran(chunk), dim=0).to(self.device)

                    # forward pass through the model
                    x_clas = self.model.forward_to_classifier(chunk_tensor)
                    x_cpool = self.model.spatial_pooling.class_wise(x_clas)

                    # scale the image to desired size
                    x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=1.0/self.extra_shrinkage)

                    # extract the central portion of the output
                    p0, p1 = self.padding_out, self.padding_out + self.window_out
                    x_cpool_ctr = x_cpool_up[:,:,p0:p1,p0:p1]

                    # place in the output
                    xout0, xout1 = u * self.window_out, ((u+1)*self.window_out)
                    yout0, yout1 = v * self.window_out, ((v+1)*self.window_out)
                    try:
                        for i in range(self.num_classes):
                            output[i, xout0:xout1,yout0:yout1] = x_cpool_ctr[0,i,:,:].cpu().detach().numpy()
                    except ValueError as e:
                        continue

                    if debug:
                        fig, axs = plt.subplots(2,2)
                        axs = axs.ravel()
                        axs[0].imshow(chunk, extent=(0,100,0,100))
                        axs[1].imshow(x_cpool_up[0,1].cpu().detach().numpy(), extent=(0,100,0,100))
                        axs[2].imshow(x_cpool_ctr[0,1].cpu().detach().numpy(), extent=(p0,p1,p0,p1))
                        axs[3].imshow(output[1])
                        plt.show()

                #
                # end of model evaluation
        #
        # end of window loop

        # cutoff the extra padded data from the windowing
        output = output[:, :self.true_out_dim[0], :self.true_out_dim[1]]

        # # print(output.shape, self.frame.img.shape, flush=True)
        # if self.debug:
        #     fig, axs = plt.subplots(1,2)
        #     axs[0].imshow(self.frame.img, extent=(0,100,0,100))
        #     axs[1].imshow(output[0], extent=(0,100,0,100))
        #     plt.show()

        return output
    #
    # end of run
#
# end of Model

class MulticlassClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiclass.dat')
        super().__init__(model_path, frame, 4)

class TangleClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tangle.dat')
        super().__init__(model_path, frame, 2, kmax=0.05, kmin=0.02, alpha=0.5, num_maps=4)

class HIPTangleClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AD_HIP_3class_5epoch_16batch_dice_lr001_v1.dat')
        super().__init__(model_path, frame, 3, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class CorticalTangleClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AD_cortical_3class_5epoch_16batch_dice_lr001_v2.dat')
        super().__init__(model_path, frame, 3, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class MicrogliaClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IBA1_microglia_6class_10epoch_16batch_dice_lr01_v0.dat')
        super().__init__(model_path, frame, 6, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class R13Classifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R13_dice_colorjitter.dat')
        super().__init__(model_path, frame, 4, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class SYN303Classifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syn303_LB.dat')
        super().__init__(model_path, frame, 2, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)
