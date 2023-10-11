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
        self.patch_raw = 224 # default: 112

        # size of window to load in
        # self.window_raw = 1120 # NOTE: wonder where this number came from?

        # padding, relative to patch_size to add to the window
        self.padding_rel = 0.5 # default: 1.0
        self.padding_raw = int(self.padding_rel * self.patch_raw) # 56

        # amount wildcat shrinks input images when mapping to segmentations
        self.wildcat_ds = 2.0

        # padding size for the output
        self.padding_out = int(self.padding_rel * self.patch_raw / self.wildcat_ds)

        # size of image to process
        self.image_size = np.array(self.frame.img.shape[:2])

        # true final patch size to output
        self.true_patch_size = self.patch_raw-self.padding_raw
        
        # dimension of the final output image
        self.out_dim = self.true_patch_size*np.ceil(self.image_size/self.true_patch_size).astype(int)
    #
    # end of constructor

    def run(self, debug=False, get_coords=False, deploy_grid=True):

        # initialize the output array
        output = np.zeros((self.num_classes, self.out_dim[0],self.out_dim[1]))

        coords = []
        if deploy_grid:
            # loop over windows
            for v in range(0, self.image_size[0], self.patch_raw-self.padding_raw): #height
                for u in range(0, self.image_size[1], self.patch_raw-self.padding_raw): #width

                    # subtract the padding
                    x0, y0 = u - (self.padding_raw//2), v - (self.padding_raw//2)
                    x1, y1 = x0 + self.patch_raw, y0 + self.patch_raw

                    xpad0, xpad1, ypad0, ypad1 = 0,0,0,0
                    if x0 < 0:
                        xpad0 = 0 - x0
                        x0 = 0
                    if y0 < 0:
                        ypad0 = 0 - y0
                        y0 = 0
                    if x1 > self.image_size[1]:
                        xpad1 = x1 - self.image_size[1]
                        x1 = self.image_size[1]
                    if y1 > self.image_size[0]:
                        ypad1 = y1 - self.image_size[0]
                        y1 = self.image_size[0]
                    
                    coords.append([(x0,y0),(x1,y1)])
                    chunk = self.frame.img[y0:y1,x0:x1]
                    chunk = np.pad(chunk, [(ypad0,ypad1),(xpad0,xpad1),(0,0)], mode='constant', constant_values=255)
                    chunk = Image.fromarray(chunk)

                    # # compute the desired size of input
                    # # TODO: input_size should be here as a ratio
                    # wwc = int(wp * self.patch_raw / self.patch_raw)

                    # resample the chunk for the two networks
                    tran = transforms.Compose([
                        # transforms.Resize((wwc, wwc)),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    # convert the read chunk to tensor format
                    with torch.no_grad():

                        # Apply transforms and turn into correct size torch tensor
                        chunk_tensor = torch.unsqueeze(tran(chunk), dim=0).to(self.device)

                        # forward pass through the model
                        x_clas = self.model.forward_to_classifier(chunk_tensor)
                        x_cpool = self.model.spatial_pooling.class_wise(x_clas)

                        # scale the image to desired size
                        x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=self.wildcat_ds)

                        # extract the central portion of the output
                        p0 = 0 + (self.padding_raw//2)
                        p1 = self.patch_raw - (self.padding_raw//2)
                        x_cpool_ctr = x_cpool_up[:,:,p0:p1,p0:p1]

                        # place in the output
                        xout0, xout1 = u, u+self.true_patch_size
                        yout0, yout1 = v, v+self.true_patch_size

                        # print()
                        # print('u,v: %d,%d' %(u,v))
                        # print('input frame shape:',self.frame.img.shape)
                        # print('x coords:', x0, x1)
                        # print('y coords:', y0, y1)
                        # print('x padded coords:',xpad0, xpad1)
                        # print('y padded coords:',ypad0, ypad1)
                        # print('x coords out:',xout0, xout1)
                        # print('y coords out:',yout0, yout1)
                        # print('center p coords:', p0, p1)
                        # print('chunk shape:',chunk._size)
                        # print('rescaled patch:',x_cpool_up.shape)
                        # print('patch shape:',x_cpool_ctr[0,:,:,:].shape)
                        # print('output patch shape:',output[:, xout0:xout1, yout0:yout1].shape)

                        output[:, yout0:yout1, xout0:xout1] = x_cpool_ctr[0,:,:,:].cpu().detach().numpy()
                        # output[:, xout0:xout1, yout0:yout1] = x_cpool_ctr[0,:,:,:].cpu().detach().numpy()

                        # exit()
                        # if debug:
                        # fig, axs = plt.subplots(2,2)
                        # axs = axs.ravel()
                        # axs[0].imshow(chunk, extent=(0,100,0,100))
                        # axs[1].imshow(x_cpool_up[0,2].cpu().detach().numpy(), extent=(0,100,0,100))
                        # axs[2].imshow(x_cpool_ctr[0,2].cpu().detach().numpy(), extent=(p0,p1,p0,p1))
                        # axs[3].imshow(output[2])
                        # plt.show()

                    #
                    # end of model evaluation
            #
            # end of window loop

            output = output[:,:self.image_size[0],:self.image_size[1]]

        # # print(output.shape, self.frame.img.shape, flush=True)
        # if self.debug:
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(self.frame.img)
        # axs[0].set_title('Orig. Frame')
        # axs[1].imshow(output[2])
        # axs[1].set_title('Model Output of tiled patches')
        # plt.show()
        else:
            frame = Image.fromarray(self.frame.img)
            # resample the chunk for the two networks
            tran = transforms.Compose([
                # transforms.Resize((wwc, wwc)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            frame_tensor = torch.unsqueeze(tran(frame), dim=0).to(self.device)

            x_clas = self.model.forward_to_classifier(frame_tensor)
            x_cpool = self.model.spatial_pooling.class_wise(x_clas)
            x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=self.wildcat_ds) #shape: [1,3,596,596]
            output = x_cpool_up[0,:,:,:].cpu().detach().numpy()

            # fig, axs = plt.subplots(1,2)
            # axs[0].imshow(self.frame.img)
            # axs[0].set_title('Orig. Frame')
            # axs[1].imshow(output[2])
            # axs[1].set_title('Full frame processed by model')
            # plt.show()

        if get_coords and deploy_grid:
            return output, coords
        else:
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

class CorticalTangleClassifier_TauC3(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AD_TauC3_cortical_5class_5epoch_16batch_dice_lr001_v3.dat')
        super().__init__(model_path, frame, 5, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class MicrogliaClassifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'microglia.dat')
        super().__init__(model_path, frame, 6, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class R13Classifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R13_dice_colorjitter.dat')
        super().__init__(model_path, frame, 4, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

class SYN303Classifier(Model):
    def __init__(self, frame):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syn303_LB.dat')
        super().__init__(model_path, frame, 2, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)
