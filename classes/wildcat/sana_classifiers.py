# TODO: change print statements to sana_logger

# system modules
import os
import time
import copy

# installed modules
import torch
import torchvision
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# custom modules
from wildcat.unet_wildcat import resnet50_wildcat_upsample

# debugging modules
from matplotlib import pyplot as plt

def get_pretrained_model(num_classes,pretrained=False,kmin=0.0,kmax=0.02,alpha=0.7,num_maps=4)
    return resnet50_wildcat_upsample(
        num_classes=num_classes,
        pretrained=pretrained,
        kmax=kmax,
        kmin=kmin,
        alpha=alpha,
        num_maps=num_maps
        )
# 
# end get_pretrained_model

def dice_loss(labels,logits,eps=1e-7):
    probas = torch.nn.functional.softmax(logits,dim=1)
    labels = labels.type(logits.type())
    # calculate over N_batch x M x M; excluding C channel
    dims = (0,2,3)
    intersection = torch.sum(probas * labels, dims)
    cardinality = torch.sum(probas + labels, dims)
    dice_score = (2. * intersection / (cardinality + eps))
    return torch.mean(1 - dice_score)
# 
# end dice_loss

def get_loss_fn():
    return dice_loss
# 
# end get_loss_criterion

def get_optimizer(model,**kwargs):
    # move these values to kwargs
    return torch.optim.SGD(model.get_config_optim(0.01, 0.1), lr=0.0001, momentum=0.9, weight_decay=1e-2)
# 
# end get_optimizer

def show(img):
    plt.imshow(img.permute(1,2,0))
    plt.gcf().set_dpi(200)
    plt.show()
# 
# end show

# check labels are being transformed w/ image properly
def display_batch(inputs,labels,class_preds,device):
    # collapse artifact and background prob maps to show in RGB space
#     class_preds = torch.nn.functional.softmax(class_preds,dim=1)
    
    # removing background from NxCxMxM output
#     class_preds = torch.cat([class_preds[:,0:1],class_preds[:,2:3],class_preds[:,3:4]],dim=1)
    
    labels = torchvision.transforms.Resize((224,224))(labels) #NxCxMxM
    class_preds = torchvision.transforms.Resize((224,224))(class_preds) #NxCxMxM
    
    plot_imgs = torch.clone(inputs).permute(0,2,3,1) #Nx3xMxM
                    
    plot_imgs = plot_imgs.to(device)
    labels = labels.to(device)
    inter_grid = np.empty((2*class_preds.shape[0],3,224,224))

    for i in range(len(inputs)):
        # color_label = torch.argmax(sum_labels,dim=0).item()
        sum_labels = torch.sum(labels[i], axis=(1,2))
        color_label = torch.nonzero(sum_labels)
        if color_label.size()[0]>0:
            if color_label[0] != color_label[-1]:
                color_label = torch.argmax(sum_labels[1:])+1
            color_label = color_label.item()
            # TODO: generalize this hip_cmap
            color = hip_cmap[color_label] #color=[r,g,b]
        else:
            color = [0,0,0]
            color_label = 0
        color = tuple(c/255 for c in color)
        
        # select 1,2,3,4 to exclude Background
        # TODO: auto-select all channels EXCEPT background
        plot_imgs[i][labels[i][[1,2,3,4]].sum(dim=0) != 0] = torch.Tensor(color).to(device)
        
        # overlay colored detection box around detection
        x = (plot_imgs[i]).cpu().detach().numpy()
        y = (inputs[i]).permute(1,2,0).cpu().detach().numpy()
        z = np.where(y==0,x,y)
        alpha = 0.3

        add_weights = cv2.addWeighted(255*x,alpha,255*z,1-alpha,0.0)
        plot_imgs[i] = torch.Tensor(add_weights)
        
        feature_img = class_preds[i][color_label:color_label+1].cpu().detach()

        # Add respective color to 1xMxM feat img, converting to 3xMxM RGB
        (r, g, b) = color
        feature_img = torch.cat([r*feature_img,g*feature_img,b*feature_img],dim=0)
        feature_img = torch.nn.functional.softmax(feature_img,dim=0)
        
        # TODO: Generalize for >3 classes 
        # superimpose feature images onto one image --> (show all class heatmaps one image for one patch)
        
        # Get 1xMxM feature image        
        bckgrnd_feat_img = class_preds[i][0:1].cpu().detach()
        NP_feat_img = class_preds[i][1:2].cpu().detach()
        tangle_feat_img = class_preds[i][2:3].cpu().detach()
        
        # Add respective color to 1xMxM feat img, converting to 3xMxM RGB
        # tangles --> red | NP --> green | background --> blue
        super_feature_img = torch.cat([tangle_feat_img, NP_feat_img, bckgrnd_feat_img],dim=0)
        super_feature_img = torch.nn.functional.softmax(super_feature_img,dim=0)  

        # add inputs/outputs to grid
        inter_grid[2*i] = (plot_imgs[i]/255).permute(2,0,1).cpu().detach().numpy()
        inter_grid[2*i+1] = (feature_img).numpy()
            
#     plt.figure(figsize=(8,8))
    plot_imgs = plot_imgs.permute(0,3,1,2)
    inter_grid = torch.tensor(inter_grid)
    grid = torchvision.utils.make_grid(inter_grid, padding=10, nrow=8, normalize=False)
    show(grid)# 
# end display_batch

# Standard training code
def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=None, device=None, **kwargs):
    if num_epochs is None:
        # TODO: do something here
        pass
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0
    phase = 'train'
    train_hist = []
    test_hist = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            batch_idx = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase],leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    x_clas = model.forward_to_classifier(inputs)
                    class_probs = model.spatial_pooling.class_wise(x_clas)
                    
#                     display_batch(inputs,labels,class_probs)
                    
                    # calculate loss
                    batch_loss = loss_fn(labels, class_probs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()
                # predict batch as most freg. pixel classification
                batch_preds = torch.argmax(torch.sum(class_probs, dim=(2,3)), dim=1)

                # looking at which class has most nonzero pixels
                batch_labels = torch.argmax(labels.count_nonzero(dim=(2,3)), dim=1)
                
                # statistics
                running_loss += batch_loss.item() * inputs.size(0)
                running_corrects += torch.sum(batch_preds == batch_labels)
                
                if batch_idx == 0:
                    display_batch(inputs,labels,class_probs,device)
                batch_idx += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Avg. Loss: {:.4f} Avg. Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
#                 test_hist.writerow(epoch_acc)
                test_hist.append((epoch_acc,epoch_loss))
            elif phase == 'train':
                train_hist.append((epoch_acc,epoch_loss))
#                 train_hist.writerow(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'train':train_hist,'test':test_hist}
# 
# end train_model

def test_model():
    pass
# 
# end test_model


class Model:
    def __init__(self, model_path, frame, num_classes, kmax=0.02, alpha=0.7, num_maps=4, kmin=0.0, debug=None):
        self.model_path = model_path
        self.frame = frame
        self.num_classes = num_classes
        self.debug = debug

        # grab the device to run the model on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load the model
        self.model = get_pretrained_model(num_classes=self.num_classes, pretrained=False,
            kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model = self.model.to(self.device)

        # size of patch to be inputted to the model
        self.patch_raw = 105

        # size of window to load in
        self.window_raw = 512

        # padding, relative to patch_size to add to the window
        self.padding_rel = 1.5
        self.padding_raw = int(self.padding_rel * self.patch_raw)

        # amount wildcat shrinks input images when mapping to segmentations
        self.wildcat_shrinkage = 2

        # don't want to store large output images
        # TODO: do we need this?
        self.extra_shrinkage = 4

        # size of output pixel (in input pixels)
        self.out_pix_size = self.wildcat_shrinkage * self.extra_shrinkage

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

        self.extra_scaled = (self.window_raw*self.n_win) / self.frame.img.shape[:-1]
        self.true_out_dim = (self.out_dim / self.extra_scaled).astype(int)
    #
    # end of constructor

    def run(self):

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
                tran = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((wwc, wwc)),
                    torchvision.transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

                    # perform the softmax to get 0->1 probs
                    x_softmax_up = scipy.special.softmax(x_cpool_up.cpu().detach().numpy(), axis=1)

                    # extract the central portion of the output
                    p0, p1 = self.padding_out, self.padding_out + self.window_out
                    x_softmax_ctr = x_softmax_up[:,:,p0:p1,p0:p1]
                    x_cpool_ctr = x_cpool_up[:,:,p0:p1,p0:p1]

                    # place in the output
                    xout0, xout1 = u * self.window_out, ((u+1)*self.window_out)
                    yout0, yout1 = v * self.window_out, ((v+1)*self.window_out)
                    try:
                        for i in range(self.num_classes):
                            # output[i, xout0:xout1,yout0:yout1] = x_softmax_ctr[0,i,:,:]
                            output[i, xout0:xout1,yout0:yout1] = x_cpool_ctr[0,i,:,:].cpu().detach().numpy()
                    except ValueError as e:
                        continue

                    # fig, axs = plt.subplots(1,3)
                    # axs[0].imshow(chunk, extent=(0,100,0,100))
                    # axs[1].imshow(x_softmax_up[0,1], extent=(0,100,0,100))
                    # axs[2].imshow(output[1])
                    # plt.show()

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
