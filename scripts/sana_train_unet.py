#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import unet
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from matplotlib import pyplot as plt
from sana_color_deconvolution import StainSeparator

torch.cuda.empty_cache()

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)



    net.train()
    return dice_score / num_val_batches

class Data(Dataset):
    def __init__(self, img_dir, mask_dir, scale=1.0):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale

        self.ids = [os.path.splitext(f)[0] \
                    for f in os.listdir(self.img_dir) if f.endswith('.png')]
        logging.info(f'Creating dataset with {len(self.ids)} files')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size

        pil_img = pil_img.resize((int(scale*w), int(scale*h)))

        img = np.asarray(pil_img)

        if img.ndim == 3 and img.shape[2] == 3:
            ss = StainSeparator('H-DAB', 'DAB', gray=True)
            img = ss.run(img)
            img = 255 - img

        if img.ndim == 2 and not is_mask:
            img = img[None, :, :]
        elif not is_mask:
            img = img.transpose((2, 0, 1))

        if not is_mask:
            img = img.astype(float) / 255
        else:
            img = img.astype(int) / 255

        return img

    @classmethod
    def load(cls, f):
        return Image.open(f)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask = os.path.join(self.mask_dir, '%s.png' % name)
        img = os.path.join(self.img_dir, '%s.png' % name)

        img = self.load(img)
        mask = self.load(mask)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

def train_net(img_dir, mask_dir, net, device, epochs=5, batch_size=1, learning_rate=0.001, val_percent=0.1,
          save_checkpoint=True, img_scale=0.5, amp=False):

    dataset = Data(img_dir, mask_dir, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(17))

    loader_args = {
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpont=save_checkpoint,
                                  img_scale=img_scale, amp=amp))
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if global_step % (n_train // (10 * batch_size)) == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(net, val_loader, device)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })
                #
                # end of evaluation
            #
            # end of batches
        #
        # end of epoch

        if save_checkpoint:
            dir_checkpoint = './unet_checkpoints'
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

def main(argv):

    # TODO: might want to train hip vs. cortical models, hip regions have much bigger neurons
    #        hip files are not all hip regions, there are cortical regions in HIP
    img_dir = r"C:\\DNPL\\data\\tangle_unet\\data\\preproc\\"
    mask_dir = r"C:\\DNPL\data\\tangle_unet\\data\\preproc_masks\\"
    epochs=5
    batch_size=4
    learning_rate=.005
    img_scale=0.5
    val_percent=0.1
    amp=False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = unet.UNet(n_channels=1, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    model_path = None
    if model_path:
        net.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f'Model loaded from {model_path}')

    net.to(device=device)
    try:
        train_net(img_dir, mask_dir,
                  net=net,
                  device=device,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  img_scale=img_scale,
                  val_percent=val_percent,
                  amp=amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
#
# end of main

if __name__ == "__main__":
    main(sys.argv)
