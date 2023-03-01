# TODO: change print statements to sana_logger
# TODO: add constructor to assemble train/test set

# system modules
import os

# installed modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as F
from PIL import Image

# custom modules
from sana_geo import Converter, Point, Polygon, plot_poly
from sana_frame import create_mask, Frame, overlay_thresh
import sana_io

# TODO: to be added in a future patch
def organize_dataset(mode,patch_root,class_map_fname,manifest_fname):
    pass
# 
# end organize_dataset

# configures initial settings for model, optimizer, and other parameters
def get_config(num_classes,
                # model params
                num_epochs=20, input_size=224, batch_size=16,
                kmin=0.0, kmax=0.02, alpha=0.7, num_maps=4,
                # optimizer params
                lr=0.001, momentum=0.9, weight_decay=1e-2,
                # affine params
                run_affine=True, affine_degrees=180, affine_translate=0.1,
                #color jitter params
                brightness=0.1, contrast=0.0, saturation=0.0, hue=0.3):

    # storing various initial params in dicts for accessibility
    config = {
    "num_classes": num_classes,
    # predefined model settings, see WildCat paper - Table 1 for more
    "model_params": {
        "num_epochs": num_epochs,
        "input_size": input_size,
        "batch_size": batch_size,
        "kmax": kmax,
        "kmin": kmin,
        "alpha": alpha,
        "num_maps": num_maps
        },
    # set options for SGD optimizer
    "optimizer_params": {
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay
        },
    "affine_params": {
        "run": run_affine,
        "degrees": affine_degrees,
        "translate": affine_translate,        
        "resnet_normalize": False #TODO: move this somewhere that makes better sense
        },
    "color_jitter_params": {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "hue": hue
        }
    }
    return config
# 
# end get_config

def get_dataset(mode, data_dir, manifest_dir, class_map_fname, config, data_transforms):
    dataset = PatchDataset(
        manifest_fname = manifest_dir,
        class_map_fname = class_map_fname,
        root = os.path.join(data_dir,mode),
        input_size = config['input_size'],
        num_classes = config['num_classes'],
        transform = data_transforms.transform,
        transform_affine = config['affine_params']
    )
    return dataset
# 
# end get_datasets

# TODO: maybe move these choices to a config['transforms'] dictionary?
def get_transforms(config,center_crop=False,to_tensor=False,color_jitter=False):
    # Transforms
    transform_list = []
    if center_crop: transform_list.append(transforms.CenterCrop(config['model_params']['input_size']//2))
    if to_tensor: transform_list.append(transforms.ToTensor())
    if config['affine_params']['resnet_normalize']: transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    if color_jitter: 
        transform_list.append(
            transforms.ColorJitter(
                brightness=config['color_jitter_params']['brightness'],
                contrast=config['color_jitter_params']['contrast'],
                saturation=config['color_jitter_params']['saturation'],
                hue=config['color_jitter_params']['hue']
                )
            )
    return transforms.Compose([transform_list])
# 
# end get_transforms

# Create WeightedRandomSampler
def get_random_sampler(dataset):
    class_counts = list(dataset.class_dist.values())
    n_samples = sum(dataset.class_dist.values())
    inv_class_freqs = [n_samples/c for c in class_counts]
    inv_class_weights = [c/sum(inv_class_freqs) for c in inv_class_freqs]
    
    # sort class names from class_mapping.json, sort them, and add them to a dictionary
    cm_dict = {}
    classnames = sorted([cm['classname'] for cm in dataset.class_map if cm['classname'] != 'ignore'])
    for i, cm in enumerate(classnames):
        cm_dict[cm] = i

    wrs_weights = []
    for i, label in enumerate(dataset.manifest['label_name']):
        for cm in dataset.class_map:
            # find the class that this label belongs to
            if label in cm['labels']:
                weight = inv_class_weights[cm_dict[cm['classname']]]
                wrs_weights.append(weight)
                
    # length of manifest, assigning weight to each sample
    # print('wrs_weights:',wrs_weights)
    # print('n_samples:',n_samples)
    return WeightedRandomSampler(wrs_weights, n_samples, replacement=True)
#
# end get_random_sampler

def get_dataloader(dataset, config, weighted_random_sample, **kwargs):
    if weighted_random_sample:
        dataloader = DataLoader(
        dataset,
        batch_size=config['model_params']['batch_size'],
        shuffle=False,
        sampler=get_random_sampler(),
        num_workers=0
        ) 
    else:    
        dataloader = DataLoader(
        dataset,
        batch_size=config['model_params']['batch_size'],
        shuffle=True,
        num_workers=0
        )
    return dataloader
# 
# end get_dataloaders

# user-defined pytorch Dataset, used to implement dice loss
class PatchDataset(Dataset):
    def __init__(self, manifest_fname, class_map_fname, root, input_size, num_classes, degrees=0, translate=0, transform=None, transform_affine=None):
        
        manifest = sana_io.load_manifest(manifest_fname)
        self.class_map = sana_io.load_class_mapping(class_map_fname)

        # remove unwanted labels from manifest
        bckgrnd_labels = [cm for cm in self.class_map if cm['classname'] == 'ignore'][0]['labels']
        for label in bckgrnd_labels:
            manifest = manifest[manifest['label_name'] != label]
        
        self.manifest = manifest

        # NOTE: assuming standard pyramidal svs storage        
        self.converter = Converter() 
        self.label_mask_lvl = 2
        
        self.root = root
        self.input_size = self.converter.to_int(Point(input_size, input_size, False, 0))
        self.num_classes = num_classes

        self.labels = sorted(os.listdir(self.root))
        self.transform = transform
        if transform_affine is not None:
            self.transform_affine = transform_affine['run']
            self.degrees = transform_affine['degrees']
            self.translate = (transform_affine['translate'], transform_affine['translate'])
        else:
            self.transform_affine = False

        self.background_idx = self.labels.index('Background')
        
        self.manifest['img_name'] = self.get_img_names(self.manifest['id'],self.manifest['label_name'])
        self.manifest = self.manifest.dropna(subset='img_name')

        self.class_dist = self.get_class_dist()

        self.build_label_masks()
    #
    # end __init__        
            
    def __len__(self):
        return len(self.manifest)
    #
    # end __len__
    
    def get_class_dist(self):
        classes = os.listdir(self.root)
        class_dist = {}
        for c in classes:
            num_imgs = len(os.listdir(os.path.join(self.root,c)))
            class_dist[c] = num_imgs
        return class_dist         
    # 
    # end get_class_dist
    
    def roi_from_corners(self,x,y,w,h):
        points = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        return Polygon(points[:,0],points[:,1],False,0)
    # 
    # end roi_from_corners

    def get_img_names(self,ids,labels,):
        img_names = []
        
        for patch_id,label in zip(ids,labels):
            label_name = [cm['classname'] for cm in self.class_map if label in cm['labels']][0]
            img_name = os.path.join(self.root, label_name, str(patch_id)+'.png')
            if os.path.exists(img_name):
                img_names.append(img_name)
            else:
                img_names.append(None)
        return img_names
    # 
    # end get_img_names

    def build_label_masks(self):

        # all the slide names in the dataset
        self.img_names = list(set(self.manifest['img_name']))

        # build the label mask for each slide
        self.label_masks = {}
        for img_name in self.img_names:

            # get all annotations in this slide
            img_df = self.manifest[self.manifest['img_name'] == img_name]

            annos = []
            mi_x, mi_y, mx_x, mx_y = np.inf, np.inf, 0, 0
            for row in img_df:

                # get the annotation name
                patch_id = row['id']
                anno_label = row['label_name']
                label_name = [cm['classname'] for cm in self.class_map if anno_label in cm['labels']][0]

                # build the ROI
                roi = self.roi_from_corners(*row[['x', 'y', 'w', 'h']].values)
                converter.rescale(roi, self.label_mask_lvl)

                if np.min(roi[:,0]) < mi_x:
                    mi_x = np.min(roi[:,0])
                if np.min(roi[:,1]) < mi_y:
                    mi_y = np.min(roi[:,1])
                if np.max(roi[:,0]) < mx_x:
                    mx_x = np.max(roi[:,0])
                if np.max(roi[:,1]) < mx_y:
                    mx_y = np.max(roi[:,1])
                
                # create an Annotation
                anno = roi.to_annotation(img_name, label_name)
                annos.append(anno)
            #
            # end of ROI creation
            
            # generate the blank mask
            padding = 128
            mask_size_x = (mx_x - mi_x) + 2*padding
            mask_size_y = (mx_y - mi_y) + 2*padding
            label_mask = Frame(np.full((mask_size_y, mask_size_x), self.background_idx), lvl=2, converter=self.converter)
            # NOTE: this is for when we don't want to assume unlabeled pixels are background
            # label_mask = Frame(np.full((mask_size_y, mask_size_x), -1), lvl=2, converter=self.converter)     

            # account for the padding in the ROIs
            # TODO: test this
            [a.translate(padding) for a in annos]

            # place each annotation in the image
            for label in self.labels:
                label_annos = [a for a in annos if a.class_name == label]
                label_idx = self.labels.index(label)
                mask = create_mask(label_annos, label_mask.size(), label_mask.lvl, label_mask.converter, x=-1, y=label_idx)
                label_mask.img[mask.img != -1] = mask.img

            fig, ax = plt.subplots(1,1)
            ax.imshow(label_mask.img)
            ax.set_title(img_name)
            plt.show()

            self.label_masks[img_name] = label_mask
        #
        # end of slides loop
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patch_id = self.manifest.iloc[idx]['id']
        anno_label = self.manifest.iloc[idx]['label_name']
        img_name = self.manifest.iloc[idx]['img_name']

        label_name = [cm['classname'] for cm in self.class_map if anno_label in cm['labels']][0]
        image = transforms.ToTensor()(Image.open(img_name).convert('RGB'))
        
        _, height, width = image.size()
        center = Point(width//2, height//2,False,0)

        # get the ROI for this sample
        roi = self.roi_from_corners(*self.manifest[['x','y','w','h']].values[idx])

        # transform ROI into the label mask coordinate system
        self.converter.rescale(roi, self.label_mask_lvl)
        roi.translate(padding) # TODO: check with the other translation!

        # get the center of the ROI in the label mask coordinates
        roi_loc, roi_size = roi.bounding_box()
        roi_center = roi_loc + roi_size//2

        # get the patch size in label mask units
        img_size = Point(width, height, False, self.label_mask_lvl)

        # get the location of the patch in the label mask
        patch_loc = self.converter.to_int(roi_center - img_size // 2 )
        patch_size = self.converter.to_int(img_size)

        # grab the portion of the label mask corresponding to this sample's patch
        mask = self.label_masks[img_name][patch_loc[1]:patch_loc[1]+patch_size[1], patch_loc[0]:patch_loc[0]+patch_size[0]]

        # resize the label mask back to the sample resolution
        mask = cv2.resize(mask, tuple(image.shape[:2][::-1]), interpolation=cv2.INTER_NEAREST)
        mask = transforms.ToTensor()(mask)
        
        # plt 1
        # fig, ax = plt.subplots(1,1)
        # ax.imshow(image.permute(1,2,0))
        # plot_poly(ax,roi)

        if self.transform_affine:
            affine = transforms.RandomAffine(
                degrees=self.degrees, translate=self.translate, fill=(255, 255, 255))
        
            affine_params = affine.get_params(
                degrees=affine.degrees,
                translate=affine.translate,
                scale_ranges=affine.scale,
                shears=affine.shear,
                img_size = (width,height)
            )
            image = transforms.functional.affine(image, *affine_params)
            mask = transforms.functional.affine(mask, *affine_params)
            
            # plt 2
            # fig, ax = plt.subplots(1,1)
            # ax.imshow(image.permute(1,2,0))
            # TODO: plot mask
            
            crop = transforms.CenterCrop((self.input_size[0], self.input_size[1]))
            image = crop.forward(image)
            mask = crop.forward(mask)
            
            # plt 3
            # fig, ax = plt.subplots(1,1)
            # ax.imshow(image.permute(1,2,0))
            # TODO: plot mask
            
        # TODO: plot the mask here, with the patch image
        pass
            
        # create the mask and the one hot label from the ROI
        # TODO: do we want non-labeled pixels as "ambiguous" aka [0,0,0,0] instead of assuming it's background
        one_hot_label = torch.zeros([self.num_classes, mask.shape[0], mask.shape[1]],dtype=int)
        for label_name in self.labels:
            label_idx = self.labels.index(label_name)
            one_hot_label[label_idx] = torch.tensor(mask == label_idx)
        one_hot_label = torchvision.transforms.Resize((112,112))(one_hot_label)

        # apply other transformations such as rotations, normalizations, colorjitter, etc.
        if self.transform is not None:
            image = self.transform(image)

        # finally, return the data!
        return image, one_hot_label
    # 
    # end __getitem__
# 
# end PatchDataset
