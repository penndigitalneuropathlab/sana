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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from PIL import Image

# custom modules
from sana_geo import Converter, Point, Polygon, plot_poly
from sana_frame import create_mask, Frame, overlay_thresh
import sana_io

def organize_dataset(mode,patch_root,class_map_fname,manifest_fname):
    pass
# 
# end organize_dataset

def get_dataset(mode, data_dir, manifest_dir, class_map_fname, config, data_transforms, degrees=180, translate=0.1):
    dataset = PatchDataset(
        manifest_fname = manifest_dir,
        class_map_fname = class_map_fname,
        root = os.path.join(data_dir,mode),
        input_size = config['input_size'],
        num_classes = config['num_classes'],
        degrees = degrees,
        translate = translate,
        transform = data_transforms[mode]
    )
    return dataset
# 
# end get_datasets

def get_data_transforms(center_crop,color_jitter):
    # TODO: how to get these options??
    # Transforms
    data_transforms = {
        'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.1,contrast=0.0,saturation=0.0,hue=0.3)                
        ]),
        'test': transforms.Compose([
        ]),
    }
    return data_transforms
# 
# end get_data_transforms

def get_dataloader(mode, manifest_dir, class_map_fname, data_dir, config, data_transforms,**kwargs):
    dataset = get_dataset(mode, manifest_dir, class_map_fname, data_dir, config, data_transforms)
    dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0
    )
    return dataloader, dataset
# 
# end get_dataloaders

class PatchDataset(Dataset):
    def __init__(self, manifest_fname, class_map_fname, root, input_size, num_classes, degrees=0, translate=0, transform=None, transform_affine=False):
        
        manifest = sana_io.load_manifest(manifest_fname)
        self.class_map = sana_io.load_class_mapping(class_map_fname)

        # remove unwanted labels from manifest
        bckgrnd_labels = [cm for cm in self.class_map if cm['classname'] == 'ignore'][0]['labels']
        for label in bckgrnd_labels:
            manifest = manifest[manifest['label_name'] != label]
        
        self.manifest = manifest

        self.root = root
        self.converter = Converter()
        self.input_size = self.converter.to_int(Point(input_size, input_size, False, 0))
        self.num_classes = num_classes

        self.labels = sorted(os.listdir(self.root))
        self.transform = transform
        self.transform_affine = transform_affine
        self.degrees = degrees
        self.translate = (translate, translate)
        self.background_idx = self.labels.index('Background')
        
        self.manifest['img_name'] = self.get_img_names(self.manifest['id'],self.manifest['label_name'])
        self.manifest = self.manifest.dropna(subset='img_name')

        self.class_dist = self.get_class_dist()
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
        
        roi = self.roi_from_corners(*self.manifest[['x','y','w','h']].values[idx])
        roi_loc, roi_size = roi.bounding_box()
        roi.translate(roi_loc-center+roi_size/2)

#         plt 1
#         fig, ax = plt.subplots(1,1)
#         ax.imshow(image.permute(1,2,0))
#         plot_poly(ax,roi)

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
            
            loc = affine_params[1]
            angle = affine_params[0]
            roi.rotate(center, -angle)
            roi.translate(-Point(loc[0],loc[1],False,0))
            
    #         plt 2
    #         fig, ax = plt.subplots(1,1)
    #         ax.imshow(image.permute(1,2,0))
    #         plot_poly(ax,roi)
            
            crop = transforms.CenterCrop((self.input_size[0], self.input_size[1]))
            image = crop.forward(image)
            
            crop_loc = center - self.input_size//2
            roi.translate(crop_loc)
            
    #         plt 3
    #         fig, ax = plt.subplots(1,1)
    #         ax.imshow(image.permute(1,2,0))
    #         plot_poly(ax,roi)
        
        # create the mask and the one hot label from the ROI
        mask = create_mask([roi], self.input_size, 0, self.converter, x=0, y=1)
        label_idx = self.labels.index(label_name)
#         print(label_idx,label_name)
        one_hot_label = torch.zeros([self.num_classes, mask.img.shape[0], mask.img.shape[1]],dtype=int) 
        one_hot_label[label_idx] = torch.tensor(mask.img.squeeze())
        one_hot_label[self.background_idx] = torch.tensor(1-mask.img.squeeze())
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