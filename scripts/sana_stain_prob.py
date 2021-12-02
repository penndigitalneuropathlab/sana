#!/usr/bin/env python

# system modules
import os
import sys
import argparse

# installed modules
# import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# custom modules
import sana_io
from sana_io import DataWriter
from sana_frame import Frame, mean_normalize, create_mask
from sana_color_deconvolution import StainSeparator
from sana_loader import Loader
from sana_geo import Point, separate_seg, plot_poly
from sana_framer import Framer

# import torch
# from torch.nn import functional as F
# from unet import UNet

# this script loads a series of slides and ROIs within the given slides
# it loads the data within ROIs, and generates a probability map representing
# the prob of each pixel containing significant data
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the slide files to process
    slides = []
    for list_f in args.lists:
        slides += sana_io.read_list_file(list_f)
    if len(slides) == 0:
        print("**> No Slides Found")
        parser.print_usage()
        exit()

    # if args.algo == 'unet':
    #     model = 'C:/DNPL/data/neuron_processing/data/unet/checkpoints/checkpoint_epoch5.pth'
    #     net = UNet(n_channels=1, n_classes=2)
    #     device = torch.device('cpu')
    #     net.to(device=device)
    #     net.load_state_dict(torch.load(model, map_location=device))

    # loop through the slides
    for slide_i, slide_f in enumerate(slides):
        print("--> Processing: %s (%d/%d)" % \
              (os.path.basename(slide_f), slide_i+1, len(slides)))

        # get the annotation file
        anno_f = sana_io.create_filepath(
            slide_f, ext='.json', fpath=args.adir, rpath=args.rdir)
        if not os.path.exists(anno_f):
            print('****> Skipping: Annotation File Not Found!')
            continue

        # initalize the Loader
        try:
            loader = Loader(slide_f)
        except:
            print('*****> Couldnt Load .svs slide')
            continue
        converter = loader.converter
        loader.set_lvl(args.lvl)

        # load the ROIs to process
        rois = sana_io.read_annotations(anno_f, args.roi_class)
        if rois is None or len(rois) == 0:
            print("****> Skipping: No ROIs Found!")
            continue

        # loop through the ROIs
        for roi_i, roi in enumerate(rois):
            print('----> Processing ROI %d/%d' % \
                  (roi_i+1, len(rois)))

            # get the prob map output filename
            out_f = sana_io.create_filepath(
                slide_f, ext=args.ofiletype, suffix='_%d_PROB' % roi_i,
                fpath=args.odir, rpath=args.rdir)

            # skip the image if already generated
            if os.path.exists(out_f):
                continue

            # initialize the data writer
            data_f = sana_io.create_filepath(
                slide_f, ext='.csv', suffix='_%d' % roi_i,
                fpath=args.odir, rpath=args.rdir)
            writer = DataWriter(data_f)
            writer.data['lvl'] = loader.lvl
            writer.data['csf_threshold'] = loader.csf_threshold

            # get the mask output filename
            mask_f = sana_io.create_filepath(
                slide_f, ext=args.ofiletype, suffix='_%d_MASK' % roi_i,
                fpath=args.odir, rpath=args.rdir)

            # scale the roi to current resolution
            converter.rescale(roi, loader.lvl)

            try:
                if args.gm_seg:
                    frame = loader.load_gm_seg(writer, roi)

                # TODO: implement this, maybe use hough, or rely on slide, or use same method as above
                elif args.crude_roi:
                    frame = loader.load_crude_roi(writer, roi)

                # no need to rotate, just load the image inside the rectangle roi
                else:
                    frame = loader.load_roi(writer, roi)
            except:
                print('****> Skipping: Cant load frame!')
                continue
            #
            # end of frame loading

            # create a binary mask to remove data outside the ROI
            mask = create_mask([roi], frame.size(), frame.lvl, frame.converter)

            # TODO: need to get this from the stain type (NeuN -> H-DAB)
            # separate the target stain from the image
            stain = 'H-DAB'
            separator = StainSeparator(stain, args.target, od=False, gray=True)
            frame = Frame(separator.run(frame.img)[0],
                          loader.lvl, loader.converter)

            # inverse image to create a stain prob map
            frame.img = 255 - frame.img

            # mask out unwanted data
            frame.mask(mask)

            # TODO: check order of operations
            # normalize image by the local means to remove
            #  inconsistent background staining
            frame = mean_normalize(loader, frame)

            # anistrophic diffusion filter to smooth the interior of objects
            frame.anisodiff()

            # TODO: maybe apply this before anisodiff??
            # opening filter to remove small objects, not considered
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
            cv.morphologyEx(frame.img, cv.MORPH_OPEN,
                            kernel=kernel, dst=frame.img)

            # TODO: IMPORTNAT: can probably decode whole image?, would want to try with scale
            #         wouldn't fit in CUDA most likely
            # TODO: instead could try to pad to avoid black bars in image, although 1kx1k is pretty big
            # TODO: try different models, maybe with RGB?
            # TODO: need to check the hyper parameters of the model
            # if args.algo == 'unet':
            #
            #     img = frame.img.astype(float) / 255
            #     img = img.transpose((2, 0, 1))
            #     net.eval()
            #     img = torch.from_numpy(img)
            #     img = img.unsqueeze(0)
            #     img = img.to(device=device, dtype=torch.float32)
            #     with torch.no_grad():
            #         output = net(img)
            #         probs = F.softmax(output, dim=1)[0][1]
            #         frame.img[:, :, 0] = (255*probs.cpu().detach().numpy()).astype(np.uint8)

                # fs = 1000
                # for i in range(0, 1+frame.img.shape[0]//fs):
                #     for j in range(0, 1+frame.img.shape[1]//fs):

                #         # get the slice bounds for the mask
                #         px1, py1, px2, py2 = 0, 0, 0, 0
                #         x1 = i*fs
                #         y1 = j*fs
                #         x2 = i*fs + fs
                #         y2 = j*fs + fs
                #         print(x1, y1, x2, y2)
                #         if x1 < 0:
                #             px1 = 0 - x1
                #             x1 = 0
                #         if y1 < 0:
                #             py1 = 0 - y1
                #             y1 = 0
                #         if x2 > frame.img.shape[1]:
                #             px2 = x2 - frame.img.shape[1]
                #             x2 = frame.img.shape[1]
                #         if y2 > frame.img.shape[0]:
                #             py2 = y2 - frame.img.shape[0]
                #             y2 = frame.img.shape[0]
                #         print(x1, y1, x2, y2, px1, py1, px2, py2)
                #         img = frame.img[y1:y2, x1:x2]
                #         print(img.shape)
                #         pady1 = np.full_like(img, 0, shape=(py1, img.shape[1], 1))
                #         pady2 = np.full_like(img, 0, shape=(py2, img.shape[1], 1))
                #         img = np.concatenate((pady1, img, pady2), axis=0)
                #         padx1 = np.full_like(img, 0, shape=(img.shape[0], px1, 1))
                #         padx2 = np.full_like(img, 0, shape=(img.shape[0], px2, 1))
                #         img = np.concatenate((padx1, img, padx2), axis=1)

                #         img = img.astype(float) / 255
                #         img = img.transpose((2, 0, 1))
                #         net.eval()
                #         img = torch.from_numpy(img)
                #         img = img.unsqueeze(0)
                #         img = img.to(device=device, dtype=torch.float32)
                #         with torch.no_grad():
                #             output = net(img)
                #             probs = F.softmax(output, dim=1)[0][1]

                #             # tf = transforms.Compose([
                #             #     transforms.ToPILImage(),
                #             #     transforms.Resize((256, 256)),
                #             #     transforms.ToTensor()
                #             # ])

                #             py2 = fs-py2
                #             px2 = fs-px2
                #             frame.img[y1:y2, x1:x2, 0] = (255*probs.cpu().detach().numpy()[:py2,:px2]).astype(np.uint8)
                #     #
                #     # end of frame_loop
                # #
                # # end of framer loop
            #
            # end of unet

            # finally, save the results
            frame.save(out_f)
            mask.save(mask_f)
            writer.write_data()
        #
        # end of frames loop
    #
    # end of slides loop
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lists', type=str, nargs='*', required=True,
                        help="filelists containing .svs files")
    # parser.add_argument('-algo', required=True, choices=['basic', 'unet'],
    #                     help="type of processing algorithm to use")
    parser.add_argument('-lvl', type=int, default=2, choices=[0, 1, 2],
                        help="specify the slide level to process on")
    parser.add_argument('-gm_seg', action='store_true')
    parser.add_argument('-crude_roi', action='store_true')
    parser.add_argument('-target', type=str, choices=['DAB', 'HEM'],
                        help="stain to process in the image")
    parser.add_argument('-roi_class', type=str, default='ROI',
                        help="class name of the ROI to process")
    parser.add_argument('-adir', type=str, default="",
                        help="location of files containing ROIs")
    parser.add_argument('-odir', type=str, default="",
                        help="location to write probability maps")
    parser.add_argument('-rdir', type=str, default="",
                        help="directory path to replace")
    parser.add_argument('-ofiletype', type=str, default='.png',
                        help="output image file extension")
    parser.add_argument('-mean_normalize', action='store_true',
                        help="")
    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
