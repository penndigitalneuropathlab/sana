#!/usr/bin/env python

# system packages
import os
import sys
import argparse

# installed packages
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
import scipy.stats as stats
import cv2
from shapely.geometry.multipolygon import MultiPolygon
from tqdm import tqdm
import pandas as pd

# custom packages
import sana_io
from sana_io import get_anno_files, read_annotations, read_list_file, create_filepath
from sana_loader import Loader
from sana_geo import Converter, plot_poly, Polygon, fix_polygon, from_shapely, transform_poly_with_params
from sana_frame import Frame, create_mask
from sana_params import Params
from sana_logger import SANALogger

# cmdl interface help messages
SLIST_HELP = "List of slides to be processed"
LVL_HELP = "Specify which pixel resolution to use while processing"
REFDIR_HELP = "Directory containing reference annotation files"
HYPDIR_HELP = "Directory containing hypothesis annotation files"
REFCLASS_HELP = "List of reference class names to use"
HYPCLASS_HELP = "List of hypothesis class names to use"
ROICLASS_HELP = "List of roi class names to load, only annotations in these rois will be considered"
IOUTHRESH_HELP = "List of IOU Thresholds to use while scoring"

# defines the location of the usage text file for cmdl interface
USAGE = os.path.join(os.environ.get('SANAPATH'),
                     'scripts', 'usage', 'sana_pr_curve.usage')

# pass two image masks to this function; pred (hyp) and true (ref) should be binary mask images 
def get_dice_coeff(ref_mask,hyp_mask):
    # intersection = 2.0 * np.sum(hyp_mask*ref_mask)
    intersection = np.sum(hyp_mask==ref_mask)
    dice = intersection / (hyp_mask.shape[0] * hyp_mask.shape[1])
    return dice

# calculates the IOU score of 2 Polygons
# NOTE: this uses Shapely, converts to shapely objects for easy calculations
def get_iou(x, y):
    # print('entered get_iou')
    # fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
    # for ref in x:
    #     plot_poly(axs[0],ref,color='red')
    # for hyp in y:
    #     plot_poly(axs[1],hyp,color='blue')

    x = x.copy().to_shapely()
    y = y.copy().to_shapely()

    x = fix_polygon(x)
    y = fix_polygon(y)

    # fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
    # for ref in x:
    #     plot_poly(axs[0],ref,color='red')
    # for hyp in y:
    #     plot_poly(axs[1],hyp,color='blue')
    # plt.show()

    i = x.intersection(y).area
    u = x.union(y).area 
    # print('i:',i)
    # print('u:',u)
    return i/u
#
# end of iou

def get_confidences(annos):
    a = annos.copy()
    confidences = np.asarray([a.confidence for a in annos])
    return confidences

def get_annotation_file(slide_f, adir, rdir, ext='.json'):
        
    # get the annotation file containing ROIs
    # anno_f = sana_io.create_filepath(
    #     slide_f, ext=ext, fpath=adir, rpath=rdir)

    # used for IBA1 microglia annotations
    anno_f = sana_io.create_filepath(
        slide_f.replace('-21_','-2021_'), ext=ext, fpath=adir, rpath=rdir)
    
    # make sure the file exists, else we skip this slide
    if not os.path.exists(anno_f):
        return None
    
    return anno_f
#
# end of get_annotation_file

# returns all annos within any of the given rois
# NOTE: if no rois given, returns all annos
def check_inside(rois, annos):

    # if no rois given, return all annos
    if len(rois) == 0:
        return annos
    else:

        # loop through the annos
        inds = []
        for i in range(len(annos)):

            # loop through the rois
            flag = False
            for roi in rois:

                # check if the anno is fully inside the roi
                # print(type(roi),type(annos[i]))
                # if annos[i].filter(roi).shape == annos[i].shape:
                if annos[i].partial_inside(roi):
                    flag = True
                    break
            #
            # end of roi loop

            if flag:
                inds.append(i)
        #
        # end of annos loop

        return [annos[i] for i in inds]
    #
    # end of roi length checking
#
# end of check_inside

def get_annos(args,load_subregion=None,get_rois=False):
    # get all annotation files in the ref and hyp dirs
    all_ref_files = sorted(get_anno_files(args.refdir))
    all_hyp_files = sorted(get_anno_files(os.path.join(args.hypdir,'detections')))

    # filter all the annotations files to remove any slides not found in the slist
    # not all the annotations found in adir are to be loaded in, it's based off contents of slist
    slist = sana_io.read_list_file(args.slist)
    files_to_load = [os.path.basename(x) for x in slist]
    if not args.sub_refdir is None and not os.path.exists(args.refdir):
        all_ref_files = sorted(get_anno_files(args.sub_refdir,recurse=True))

    overlap_slides = []
    if not args.sub_refdir is None:
        all_ref_files = []
        seen_slides = []
        anno_folders = os.listdir(args.sub_refdir)
        for anno_fldr in anno_folders:
            anno_path = os.path.join(args.sub_refdir,anno_fldr)
            anno_files = os.listdir(anno_path)            
            process_anno_files = [a for a in slist if os.path.basename(a).replace('.svs','.json') in anno_files]
            for a_file in process_anno_files:
                if os.path.basename(a_file) in seen_slides:
                    overlap_slides.append(os.path.basename(a_file))
                else:
                    seen_slides.append(os.path.basename(a_file))
                    anno_file_path = os.path.join(anno_path,os.path.basename(a_file).replace('.svs','.json'))
                    all_ref_files.append(anno_file_path)
                    
    ref_files = [x for x in all_ref_files if os.path.basename(x).replace('.json','.svs') in files_to_load]
    hyp_files = [x for x in all_hyp_files if os.path.basename(x).replace('.json','.svs') in files_to_load]
        
    if not len(ref_files):
        print('No ref files found...')
        exit()
    if not len(hyp_files):
        print('No hyp files found...')
        exit()

    # loop through annotation files
    ref_annos, hyp_annos, roi_annos = [], [], []
    # for rf in tqdm(ref_files):
    for rf in ref_files:
        for hf in hyp_files:
            # make sure there is a one to one match between ref and hyp dirs
            if os.path.basename(rf) == os.path.basename(hf):

                # load the ref rois, if given
                rois = []
                if not args.roiname is None:
                    if not args.roiclass is None:
                        for roi_class in args.roiclass:
                            rois += read_annotations(rf, name=args.roiname, class_name=roi_class)
                    elif not load_subregion is None:
                        rois += read_annotations(rf, name=args.roiname, class_name=load_subregion)
                    elif not args.roisubregion is None:
                        for roisubregion in args.roisubregion:
                            rois += read_annotations(rf, name=args.roiname, class_name=roisubregion)
                    else:
                        rois += read_annotations(rf, name=args.roiname)
                else:
                    print('No roi name given, please enter a target name')
                #
                # end of roi reading
                # print('len(loaded rois):',len(rois))

                if len(rois) == 0:
                    print('Could not read ROI annos...%s' %os.path.basename(rf))
                    continue


                if os.path.basename(rf).replace('.json','.svs') in overlap_slides:
                    annotators = []
                    for i, anno_fldr in enumerate(os.listdir(args.sub_refdir)):
                        anno_f = os.path.join(args.sub_refdir,anno_fldr,os.path.basename(rf))
                        if os.path.exists(anno_f):
                            annotators.append(anno_fldr)
                    chance = np.random.randint(len(annotators))
                    rf = os.path.join(args.sub_refdir,annotators[chance],os.path.basename(rf))

                    # load the ref annotations with any of the given ref classes
                    if args.refname is None and args.refclass is None:
                        print('loading all REF annos...')
                        ra = read_annotations(rf)
                    elif args.refname is None and not args.refclass is None:
                        print('loading REF class...')
                        ra = []
                        for rclass in args.refclass:
                            # print(rclass)
                            ra += read_annotations(rf, class_name=rclass)
                    elif not args.refname is None and args.refclass is None:
                        print('loading REF name...')
                        ra = []
                        for rname in args.refname:
                            ra += read_annotations(rf, name=rname)

                    #
                    # end of anno_fldr loop
                   

                else:
                    # load the ref annotations with any of the given ref classes
                    if args.refname is None and args.refclass is None:
                        print('loading all REF annos...')
                        ra = read_annotations(rf)
                    elif args.refname is None and not args.refclass is None:
                        print('loading REF class...')
                        ra = []
                        for rclass in args.refclass:
                            # print(rclass)
                            ra += read_annotations(rf, class_name=rclass)
                    elif not args.refname is None and args.refclass is None:
                        print('loading REF name...')
                        ra = []
                        for rname in args.refname:
                            ra += read_annotations(rf, name=rname)
                #
                # end of ref anno reading
                # print('len(loaded ref):',len(ra))

                
                if len(ra) == 0:
                    print('Could not read REF annos...%s' %os.path.basename(rf))
               

                # load the hyp annotations and confidences with a given hyp class
                if args.hypclass is None:
                    ha = read_annotations(hf)
                else:
                    ha = []
                    # print(args.hypclass)
                    for hclass in args.hypclass:
                        # print(hclass)
                        ha += read_annotations(hf, class_name=hclass)
                #
                # end of hyp anno reading
                # print('len(loaded hyp):',len(ha))

                # filter hyp annos by confidence threshold
                ha = [h for h in ha if h.confidence >=1.42]

                if len(ha) == 0:
                    print('Could not read HYP annos...%s' %os.path.basename(hf))

                # keep only the annotations within a given roi
                
                # Plot ra's and ha's before filtering by rois
                # slides = sana_io.read_list_file(args.slist)
                # img_dir = [s for s in slides if os.path.basename(s) == os.path.basename(hf).replace('.json','.svs')][0]

                # if len(img_dir)==0:
                #     continue
                # # initialize loader, this will allow us to load slide data into memory
                # image = sana_io.create_filepath(img_dir)

                # loader = Loader(image)
                # loader.set_lvl(0)
                # thumb_frame = loader.load_thumbnail()

                # conv = Converter()
                # for ROI in rois:
                #     if ROI.lvl != thumb_frame.lvl:
                #         conv.rescale(ROI, thumb_frame.lvl)
                # for REF_ANNO in ra:
                #     if REF_ANNO.lvl != thumb_frame.lvl:
                #         conv.rescale(REF_ANNO, thumb_frame.lvl)
                # for HYP_ANNO in ha:
                #     if HYP_ANNO.lvl != thumb_frame.lvl:
                #         conv.rescale(HYP_ANNO, thumb_frame.lvl)

                # fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
                # axs = axs.ravel()
                # fig.suptitle('check_inside ROI')
                # for ROI in rois:
                #     # Plotting the processed frame and processed frame with clusters
                #     axs[0].imshow(thumb_frame.img, cmap='gray')
                #     axs[0].set_title('Ref Before Check | %d polys' %len(ra))
                #     for REF_ANNO in ra:
                #         plot_poly(axs[0], REF_ANNO, color='red')

                #     axs[1].imshow(thumb_frame.img, cmap='gray')
                #     axs[1].set_title('Hyp Before Check | %d polys' %len(ha))
                #     for HYP_ANNO in ha:
                #         plot_poly(axs[1], HYP_ANNO, color='blue')
                tile_refs = check_inside(rois, ra)
                # print('--ref_annos:',len(ref_annos))
                # print('hyp:')
                tile_hyps = check_inside(rois, ha)
                # print('--hyp_annos:',len(hyp_annos))
                # print()

                
                # for ROI in rois:
                #     # Plotting the processed frame and processed frame with clusters
                #     axs[2].imshow(thumb_frame.img, cmap='gray')
                #     axs[2].set_title('Ref After Check | %d polys' %len(tile_refs))
                #     for REF_ANNO in tile_refs:
                #         plot_poly(axs[2], REF_ANNO, color='red')

                #     axs[3].imshow(thumb_frame.img, cmap='gray')
                #     axs[3].set_title('Hyp After Check | %d polys' %len(tile_hyps))
                #     for HYP_ANNO in tile_hyps:
                #         plot_poly(axs[3], HYP_ANNO, color='blue')
                # plt.show()

                ref_annos += tile_refs
                hyp_annos += tile_hyps
                roi_annos += rois
                # ref_annos += ra
                # hyp_annos += ha              
            #
            # end of file matching check
        #
        # end of hyp file loop
    #
    # end of ref file loop

    if not len(ref_annos):
        print('Could not load ref annos...')
        exit()
    if not len(hyp_annos):
        print('Could not load hyp annos...')
        exit()
    
    # sort the hyp annotations from most to least confident
    # hyp_annos = hyp_annos[::2]
    hyp_annos.sort(key=lambda x: x.confidence, reverse=True)

    print('Loaded in %d ROI annotations' %len(roi_annos))
    print('Loaded in %d Reference annotations' %len(ref_annos))
    print('Loaded in %d Hypothesis annotations' %len(hyp_annos))
    print()
    if get_rois:
        return ref_annos, hyp_annos, roi_annos
    else:
        return ref_annos, hyp_annos
#
# end of get_annos

def error_analyze(all_ref, all_hyp, all_rois, iou_threshold, args, load_subregion=None):
    warning = input('DO NOT RUN ON BLIND DATASET, ARE YOU SURE YOU WANT TO CONTINUE? (Y/N)')
    if warning.lower() in ('yes','y'):
        img_dir = sana_io.read_list_file(args.slist)
        all_slides = [os.path.basename(x) for x in img_dir]
        if not load_subregion is None:
            inside_annos_df_fname = os.path.join(args.odir,'Subregions','inside_annos_subregions.csv')
        else:
            inside_annos_df_fname = os.path.join(args.odir,'IoUs','inside_annos_ious.csv')
        
        df_columns = [
            'slide_name',
            'tile_class',
            'tile_name',
            'n_ref',
            'n_hyp'
        ]

        inside_annos_df = pd.DataFrame(columns=df_columns)

        total_error_pr = {
            'precision': [],
            'recall': [],
            'auc': []
        }

        
        for slide_name in all_slides:
            tp_fp_annos = []
            first_run_tp = True
            # print(slide_name.replace('.svs',''))
            # slide_name = 2012-116-30F_R_MFC_R13_25K_05-23-19_DC.svs

            # build path name to get _Tile *'s
            # TODO: move this to a sana_io.get_output_file(slide_name,odir,suffix='_ORIG',extension='.json')
            # print(slide_name)
            # bid = sana_io.get_bid(slide_name)
            # antibody = sana_io.get_antibody(slide_name)
            # slide_region = sana_io.get_region(slide_name)
            # odir = os.path.join(args.hypdir, bid, antibody, region)
            roi_dir = os.path.join(args.hypdir,slide_name.replace('.svs',''))
            
            if not os.path.exists(roi_dir):
                continue
            main_roi_dirs = [ f.path for f in os.scandir(roi_dir) if f.is_dir() and f.name.split('_')[0] in args.roiclass ]
            
            # grab annos with same slide name
            slide_refs = [x for x in all_ref if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
            slide_hyps = [x for x in all_hyp if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
      
            # print('len(slide_refs):',len(slide_refs))
            # print('len(slide_hyps):',len(slide_hyps))

            # loop through ROIs in a slide
            for i, main_roi_dir in enumerate(main_roi_dirs):
                main_roi_name = os.path.split(main_roi_dir)[1]
                # ex. tile_class = GM
                # ex. tile_name = Tile 52
                tile_class, tile_name = main_roi_name.split('_',1)
                # print('Main ROI:',tile_class, '|' ,tile_name)
                
                # fetch main_roi based on class names and tile names (if exists)
                main_roi = [x for x in all_rois if (x.class_name == tile_class) and (x.name == tile_name) and (os.path.basename(x.file_name).replace('.json','.svs')==slide_name)]
                # main_roi = [x for x in main_rois if (x.class_name == tile_class)]
                # filter by subregion if doing subregion analysis
                if not load_subregion is None:
                    print('Separating subregion ROIs - %s' %load_subregion)
                    main_roi = [x for x in main_roi if x.class_name == load_subregion]

                if len(main_roi)==0:
                    # print('No main rois...')
                    continue
                elif len(main_roi)>1:
                    [print(os.path.basename(a.file_name)+' | '+a.name+' | '+a.class_name) for a in main_roi]
                else:
                    main_roi = main_roi[0]

                # print(main_roi.name,main_roi.class_name)
                

                params_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','.csv'))
                params = Params(params_fname)
                padding = params.data['padding']
                
                conv = Converter()
                orig_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_ORIG.png'))
                orig_frame = Frame(orig_fname, lvl=0, converter=conv)

                # w, h = params.data['size']
                # x = params.data['loc'][0]
                # y = params.data['loc'][1]
                # roi_sq = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                # main_roi = Polygon(roi_sq[:,0],roi_sq[:,1],is_micron=False,lvl=main_roi.lvl)
                
                roi_plot = main_roi.copy()
                roi_plot.translate(params.data['loc'])
                # print(params.data['loc'])

                # roi_loc, _ = roi_plot.bounding_box()
                # roi_loc = roi_loc-(padding//2,padding//2)
                # roi_plot.translate(roi_loc)
                
                # ref, hyp = ...
                ref = [x for x in slide_refs if x.partial_inside(main_roi)]  
                hyp = [x for x in slide_hyps if x.partial_inside(main_roi)]
                num_ref_inside = len(ref)
                num_hyp_inside = len(hyp)

                # print('slide_name:',slide_name)
                # print('main_roi_name:',main_roi_name)
                # print(main_roi.file_name)
                # print('inside refs:',num_ref_inside)
                # print('inside hyps:',num_hyp_inside)
                # print()

                # TODO: rename results/seen 
                results, seen, ref, hyp = iou_score(ref, hyp, iou_threshold)

                cumulative_tp = np.cumsum(results)

                # calculate rolling precision
                # prec = TP / Num. Detections
                precision = cumulative_tp / np.arange(1, len(hyp)+1)

                # calculate rolling recal
                # rec = TP / Num. Positives
                recall = cumulative_tp / len(ref)

                # integrate to get area under the curve
            
                total_error_pr['precision'] += precision.tolist()
                total_error_pr['recall'] += recall.tolist()                
                
                # only do the following if there are errors!
                # TODO: this should stop it from plotting the 100% accurate ROIs, but always run this code otherwise
                if (False in results) or (False in seen):
                    plot_errors = True
                    # print('plot check')
                else:
                    plot_errors = False

                # add error info to csv
                row = {
                    'slide_name': [slide_name],
                    'tile_class': [tile_class],
                    'tile_name': [tile_name],
                    'n_ref': [num_ref_inside],
                    'n_hyp': [num_hyp_inside]
                }

                inside_anno_row = pd.DataFrame(row)
                inside_annos_df = pd.concat([inside_annos_df,inside_anno_row], ignore_index = True)
                
                # use sana_io.create_filepath function instead of this os.path.join
                wc_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_PROBS.npy'))
                try:
                    wc_activation = np.load(wc_fname)
                    wc_softmax = softmax(wc_activation,axis=0)
                except:
                    continue
                
                # # LB detections are at dim 1 in wc_activations (and nonsoftmax'd)
                # lb_activation = np.rint(255*wc_softmax[1,:,:]).astype(np.uint8)
                # lb_activation = cv2.resize(lb_activation, orig_frame.size(),interpolation=cv2.INTER_LINEAR)
                # lb_activation = lb_activation[:,:,None]
                tangle_idx = 2

                lb_softmax = np.rint(255*wc_softmax[tangle_idx,:,:]).astype(np.uint8)
                lb_softmax = cv2.resize(lb_softmax, orig_frame.size(),interpolation=cv2.INTER_LINEAR)
                lb_softmax = lb_softmax[:,:,None]
                
                # # removing padding from original image and wildcat greyscale
                # orig_frame.img = orig_frame.img[padding//2:-padding//2,padding//2:-padding//2]
                # lb_softmax = lb_softmax[padding//2:-padding//2,padding//2:-padding//2]
                # [x.translate(padding//2+params.data['loc']) for x in ref]
                # [x.translate(padding//2+params.data['loc']) for x in hyp]

                [x.translate(params.data['loc']) for x in ref]
                [x.translate(params.data['loc']) for x in hyp]
                # [x.translate(-roi_plot.bounding_box()[0]) for x in ref]
                # [x.translate(-roi_plot.bounding_box()[0]) for x in hyp]

                alpha = np.full_like(lb_softmax, 60)
                # get probs > 30%
                alpha[lb_softmax<255*args.softmax_thresh] = 0
                z = np.zeros_like(lb_softmax)

                rgba_wc = np.concatenate([z,lb_softmax,z,alpha],axis=2)

                if len(ref)>1:
                    # print(os.path.normpath(ref[0].file_name).split(os.sep)[-2])
                    annotator = os.path.normpath(ref[0].file_name).split(os.sep)[-2].split('_')[1]
                    # print(annotator)
                    # exit()
                else:
                    annotator = '__'
                
                # Plot Hit/TPs and Miss/FPs overlay
                if plot_errors:
                    fig, ax = plt.subplots(1,1)
                    fig.suptitle('WildCat Probability of Tangle Detection >%d%%\n%s\nAnnotator: %s | %s-%s \n IoU: %0.1f | Polygon Count: Ref: %d / Hyp: %d'
                                    %(int(args.softmax_thresh*100),slide_name.replace('.svs',''),annotator,tile_class,tile_name,iou_threshold,num_ref_inside,num_hyp_inside))
                
                    ax.imshow(orig_frame.img)
                    ax.imshow(rgba_wc)
                    plot_poly(ax,roi_plot,color='red')
                
                # True in results is green, label=TP
                # False in results is red, label=FP
                plot_tp = True
                plot_fp = True
                for h, poly in enumerate(hyp):
                    if results[h] == True:
                        if plot_errors:
                            plot_poly(ax,poly,label='Hyp. TP | Auto' if plot_tp else '', color='limegreen', linewidth=1.0)
                            plot_tp = False
                        fname = slide_name.replace('.svs','')
                        tp_anno = poly.to_annotation(
                            fname,
                            anno_name = tile_name,
                            class_name = 'TP'
                        )
                        # tp_anno.translate(-params.data['loc'])
                        transform_poly_with_params(tp_anno,params,inverse=True)
                        tp_fp_annos.append(tp_anno)

                    if results[h] == False:
                        if plot_errors:
                            plot_poly(ax,poly,label='Hyp. FP | Auto' if plot_fp else '', color='yellow', linewidth=.5)
                            plot_fp = False
                        fname = slide_name.replace('.svs','')
                        fp_anno = poly.to_annotation(
                            fname,
                            anno_name = tile_name,
                            class_name = 'FP'
                        )
                        # fp_anno.translate(-params.data['loc'])
                        transform_poly_with_params(fp_anno,params,inverse=True)
                        tp_fp_annos.append(fp_anno)
                        
                
                # True in seen is blue, label=hit
                # False in seen in orange, label=miss                     
                plot_hit = True
                plot_miss = True
                for r, poly in enumerate(ref):
                    if seen[r] == True and plot_errors:
                        plot_poly(ax,poly,label='Ref. Hit | Manual' if plot_hit else '', color='blue', linewidth=.5)
                        plot_hit = False
                    if seen[r] == False and plot_errors:
                        plot_poly(ax,poly,label='Ref. Miss | Manual' if plot_miss else '', color='orange', linewidth=.75)
                        plot_miss = False
                if plot_errors:
                    # Shrink current axis by 20%
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)

                    fig.tight_layout()
                    plt.tick_params(
                        axis='both',       # changes apply to both axes
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        left=False,        # ticks along the left edge are off
                        labelbottom=False, # labels along the bottom edge are off
                        labelleft=False    # labels along the left edge are off
                    )
                # don't show the plot, save to some directory (args.odir)
                if not load_subregion is None:
                    fig_fname = os.path.join(args.odir,'Subregions','error_analysis',load_subregion,slide_name.replace('.svs','_IOU_%0.1f_ERRORS_%d.png' %(iou_threshold, i)))
                else:
                    fig_fname = os.path.join(args.odir,'IoUs','error_analysis',str(iou_threshold),slide_name.replace('.svs','_IOU_%0.1f_ERRORS_%d.png' %(iou_threshold, i)))
                
                if plot_errors:
                    sana_io.create_directory(fig_fname)
                    plt.savefig(fig_fname)
                    # plt.show()
                    plt.close()
                # 
                # end of TP/FP overlay plot
                #
                # end of results/seen check
            #
            # end of main_roi loop

            # print('slide hyps:',len(slide_hyps))
            # print('TP FP annos:',len(tp_fp_annos))
            print('Error analysis performed...')

            # TP/FP annotations prep
            # tp_annos = [transform_poly_with_params(tp,params,inverse=True) for tp in tp_annos]
            # fp_annos = [transform_poly_with_params(fp,params,inverse=True) for fp in fp_annos]

            # TODO: just write out the hyp annos to a new file, with the x.anno_name editted to be TP or FP

            [orig_frame.converter.rescale(tp_fp_a,0) for tp_fp_a in tp_fp_annos]

            tp_fp_file = slide_name.replace('.svs','_TP_FPs.json')
            tp_fp_odir = os.path.join(args.odir,'IoUs','tp_fp_auto_annos',str(iou_threshold),'TP_FPs')
            os.makedirs(tp_fp_odir,exist_ok=True)
            
            tp_fp_anno_fname = os.path.join(tp_fp_odir,tp_fp_file)

            if not os.path.exists(tp_fp_anno_fname) or first_run_tp:
                sana_io.write_annotations(tp_fp_anno_fname,tp_fp_annos)
                first_run_tp = False
            else:
                sana_io.append_annotations(tp_fp_anno_fname,tp_fp_annos)

        #
        # end of slide_name loop
    else:
        print('Error analysis NOT performed...')
    #
    # end of warning y/n check
# 
# end of error_analyze
            

# NOTE: inputs must be a list of annotations that have the same coords in slide coord. system
# this function takes in two lists of annotations and turns them into binary masks to then calculate dice score
def dice_score(roi_anno, ref_annos, hyp_annos):
    
    # convert annotations to binary masks
    roi_anno = roi_anno[0]
    loc, size = roi_anno.bounding_box()

    roi_anno.translate(loc)
    [r.translate(loc) for r in ref_annos]
    [h.translate(loc) for h in hyp_annos]
    
    
    converter = Converter()

    # create tile binary masks
    ref_mask = create_mask(ref_annos,size=size,lvl=0,converter=converter)
    hyp_mask = create_mask(hyp_annos,size=size,lvl=0,converter=converter)

    ref_mask.img = np.flip(ref_mask.img,axis=0)
    hyp_mask.img = np.flip(hyp_mask.img,axis=0)

    # fig, axs = plt.subplots(1,2)
    # plot_poly(axs[0],roi_anno,color='green')
    # plot_poly(axs[1],roi_anno,color='green')
    # [plot_poly(axs[0],r_anno,color='red') for r_anno in ref_annos]
    # [plot_poly(axs[1],h_anno,color='blue') for h_anno in hyp_annos]
    # axs[0].set_title('Ref')
    # axs[1].set_title('Hyp')
    # fig.tight_layout()
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(ref_mask.img,cmap='gray')
    # axs[1].imshow(hyp_mask.img,cmap='gray')
    # plt.show()

    return get_dice_coeff(ref_mask.img,hyp_mask.img)
#
# end of dice_score
            


# NOTE: this can be any subset of a dataset! works on multiple slides and ROIs
def iou_score(ref_annos, hyp_annos, iou_threshold):
    print('Scoring annotations...')

    # array of trues/falses denoting if each reference has been used up in the scoring
    # NOTE: True denotes that this annotation was hit, False means it was missed
    seen = np.zeros(len(ref_annos), dtype=bool)

    # array of trues/falses denoting if each hypothesis is a TP or FP
    results = np.zeros(len(hyp_annos), dtype=bool)

    # all FN's or all FP's if either list is empty
    if len(hyp_annos)==0 or len(ref_annos)==0:
        return results, seen, ref_annos, hyp_annos

    # loop through the hypothesis annotations from most to least confident
    hyp_annos = sorted(hyp_annos, key=lambda x: x.confidence, reverse=True)
    # for hyp_i, hyp in enumerate(tqdm(hyp_annos)):
    for hyp_i, hyp in enumerate(hyp_annos):

        # calculate the IOU scores between all refs and the current hyp
        iou_scores = np.zeros(len(ref_annos), dtype=float)
        for ref_i, ref in enumerate(ref_annos):

            # only compare ref and hyp if they are from the same file!
            if os.path.basename(ref.file_name) == os.path.basename(hyp.file_name):
                try:
                    iou_scores[ref_i] = get_iou(ref, hyp)
                except Exception as e:
                    # TODO: what to do about shapely failing here??
                    # TODO: can we write our own calculate_iou()?
                    print(e)
                    iou_scores[ref_i] = 0
                    r = ref.copy()
                    h = hyp.copy()
                    
                    fig, axs = plt.subplots(1,2)                    
                    fig.suptitle('Before fix_poly')
                    plot_poly(axs[0],r)
                    plot_poly(axs[1],h)

                    r = from_shapely(fix_polygon(r.to_shapely()))
                    h = from_shapely(fix_polygon(h.to_shapely()))

                    # print('Ref:',r)
                    # print('Hyp:',h)
                    fig, axs = plt.subplots(1,2)                    
                    fig.suptitle('After fix_poly')
                    plot_poly(axs[0],r)
                    plot_poly(axs[1],h)
                    plt.show()
                    # poly_fname = 'C:/Users/eteun/dev/data/lewy_pilot/bad_polys'
                    # bad_polys = os.listdir(poly_fname)
                    # bad_poly_name = os.path.basename(hyp.file_name).replace('.json','_bad_poly_%d.npy' %len(bad_polys))
                    # np.save(os.path.join(poly_fname,bad_poly_name),hyp)
        #
        # end of iou calculation

        # get the max IOU between ref's and the current hyp
        best_match = np.argmax(iou_scores)

        # current hyp matches with a ref, potential TP
        # TODO: should this look for all refs with high IOU? important for the
        #       case if a detection is overlapping w/ multiple refs
        if iou_scores[best_match] >= iou_threshold:

            # ref not used yet in the scoring, TP
            if not seen[best_match]:
                results[hyp_i] = True
                seen[best_match] = True

            # ref was already used with another hyp, FP
            else:
                results[hyp_i] = False
            #
            # end of ref seen checking
        #
        # end of iou threshold
    #
    # end of hyp scoring

    return results, seen, ref_annos, hyp_annos
#
# end of iou_score

def pr(ref_annos, hyp_annos, iou_threshold, args):
    print('Running Precision-Recall calculation...')

    # get the TP, FP, hits, misses, and sorted by confidence hyp annos
    results, seen, ref_annos, hyp_annos = iou_score(ref_annos, hyp_annos, iou_threshold)

    # calculate rolling sum of TP's
    # NOTE: this is from most to least confident
    cumulative_tp = np.cumsum(results)

    # calculate rolling precision
    # prec = TP / Num. Detections
    precision = cumulative_tp / np.arange(1, len(hyp_annos)+1)

    # calculate rolling recal
    # rec = TP / Num. Positives
    recall = cumulative_tp / len(ref_annos)

    # integrate to get area under the curve
    midpoints = (precision[1:] + precision[:-1]) / 2
    widths = recall[1:] - recall[:-1]
    ap = np.sum(midpoints * widths)

    return precision, recall, ap, results, seen
#
# end of pr

# returns the ref and hyp annotations inside a given ROI from a slide
def get_tile_annos(slide,roi,args):
    
    ref_anno_fname = os.path.join(args.refdir,slide)
    for rclass in args.refclass:
        ref_slide_annos = sana_io.read_annotations(ref_anno_fname,class_name=rclass)
    
    hyp_anno_fname = os.path.join(args.hypdir,'detections',slide)
    for hclass in args.hypclass:
        hyp_slide_annos = sana_io.read_annotations(hyp_anno_fname,class_name=hclass)

    ref_tile_annos = check_inside(roi,ref_slide_annos)
    hyp_tile_annos = check_inside(roi,hyp_slide_annos)

    return ref_tile_annos, hyp_tile_annos

    

def get_dice_hist(args,logger=None):
    print('Generating histogram from Dice Coeff.\'s...')
    
    process_list = read_list_file(args.slist)
    slides = [os.path.basename(s) for s in process_list]
    slides = [s for s in slides if s.replace('.svs','.json') in os.listdir(os.path.join(args.hypdir,'detections'))]

    dice_scores = []
    ref_tile_counts = []
    hyp_tile_counts = []
    ref_anno_count = 0
    hyp_anno_count = 0
    error_tile_names = []
    good_tile_names = []
    min_dice_score = 1.0
    # loop through slides found in hyp dir
    for slide in tqdm(slides):
        slide_anno_fname = os.path.join(args.refdir,slide.replace('.svs','.json'))

        slide_roi_annos = []
        # load ROIs in slide
        for roiclass in args.roiclass:
            slide_roi_annos += sana_io.read_annotations(slide_anno_fname,name=args.roiname,class_name=roiclass)

        tile_names = list(set([roi.name for roi in slide_roi_annos]))

        # loop through tile names in slide
        for tile in tile_names:

            # get tile ROI annotation from list of slide ROIs
            roi_anno = [roi for roi in slide_roi_annos if roi.name==tile]

            # get manual and SANA annotations inside tile
            ref_tile_annos, hyp_tile_annos = get_tile_annos(slide.replace('.svs','.json'),roi_anno,args)
            
            tile_dice_score = dice_score(roi_anno,ref_tile_annos,hyp_tile_annos)
            
            if tile_dice_score < min_dice_score:
                min_dice_score = tile_dice_score

            tile_fname = os.path.basename(roi_anno[0].file_name).replace('.json','')
            if tile_dice_score <= 0.984:
                error_tile_names.append((tile_fname+'-'+roi_anno[0].name+'-%0.3f') %tile_dice_score)
            elif tile_dice_score > 0.85:
                good_tile_names.append((tile_fname+'-'+roi_anno[0].name+'-%0.3f') %tile_dice_score)


            dice_scores.append(tile_dice_score)
            ref_tile_counts.append(len(ref_tile_annos))
            hyp_tile_counts.append(len(hyp_tile_annos))

            ref_anno_count += len(ref_tile_annos)
            hyp_anno_count += len(hyp_tile_annos)
    
    with open(os.path.join(args.odir,'error_tiles.list'),'w') as f:
        [f.write(i+'\n') for i in error_tile_names]
    with open(os.path.join(args.odir,'good_tiles.list'),'w') as f:
        [f.write(i+'\n') for i in good_tile_names]

    # finally, show the plot
    fig, ax = plt.subplots(1,1,figsize=(14,8))
    step = 0.005
    nbins = np.arange(np.floor(np.min(dice_scores)*100.0)/100.0,1.01,step=step)
    n, bins, patches = ax.hist(dice_scores,bins=nbins,rwidth=0.9)
            
    ax.set_title('Dice Coefficients of Tile\'s\nRef Tile Count: %d | Hyp Tile Count: %d' %(ref_anno_count,hyp_anno_count))
    ax.set_xlabel('Dice Coeff.')
    ax.set_ylabel('Counts')
    ax.set_xticks(bins)
    plt.savefig(os.path.join(args.odir,'dice_hist.png'), dpi=300)
    print('PNG saved:',args.odir)
    
    
    fig, ax = plt.subplots(1,1)
    nbins = np.arange(np.min(dice_scores),np.max(dice_scores),0.1)
    dice_density = [dice_scores[i]/ref_tile_counts[i] for i in range(len(dice_scores))]
    n, bins, patches = ax.hist(dice_density,bins=nbins,rwidth=0.9)
    # n, bins, patches = ax.hist([dice_scores[i]/ref_tile_counts[i] for i in range(len(dice_scores))],bins=nbins,rwidth=0.9,alpha=0.5,label='Ref Annos')
    # n, bins, patches = ax.hist([dice_scores[i]/hyp_tile_counts[i] for i in range(len(dice_scores))],bins=bins,rwidth=0.9,alpha=0.5,label='Hyp Annos')
    ax.set_title('Dice Coefficients/Tile Counts of Tile\'s\nRef Tile Count: %d | Hyp Tile Count: %d' %(ref_anno_count,hyp_anno_count))
    ax.set_xlabel('Dice Coeff./Ref Tile Tangle Count')
    ax.set_ylabel('Counts')
    ax.set_xticks(bins)
    # plt.legend()
    plt.savefig(os.path.join(args.odir,'dice_score_density.png'),dpi=300)

    fig, ax = plt.subplots(1,1)
    ax.set_title('Tangle Count vs Dice Coeff.')
    ax.scatter(dice_scores,ref_tile_counts)
    # ax.scatter(dice_scores,ref_tile_counts,label='Ref Annos')
    # ax.scatter(dice_scores,hyp_tile_counts,label='Hyp Annos')
    ax.set_xlabel('Dice Coeff.')
    ax.set_ylabel('Ref Tangle Count')
    ax.set_xticks(bins)
    # plt.legend(loc='upper left')
    plt.savefig(os.path.join(args.odir,'dice_score_v_counts.png'),dpi=300)
    # plt.show()
    
# 
# end of get_iou_pr


def get_iou_pr(args,logger=None):
    print('Generating PR curve for IoUs...')
    ref_annos, hyp_annos, roi_annos = get_annos(args,get_rois=True)

    if ref_annos and hyp_annos:
        print('Annotations loaded...')

    # fig, ax = plt.subplots(1,1)
    # loop through all provided iou thresholds
    pr_dict = {
        'precision': [],
        'recall': [],
        'auc': []
    }
    prev_results = 0
    prev_seen = 0
    for iou_threshold in args.iouthresh:
        if args.analyze_errors:
            error_analyze(ref_annos,hyp_annos,roi_annos,iou_threshold,args)
        
        # generate precision/recall curve, calculate the average precision
        precision, recall, auc, results, seen = pr(ref_annos, hyp_annos, iou_threshold, args)
        confidences = get_confidences(hyp_annos)

        # plot confidences
        fig, axs = plt.subplots(1,1)
        axs.hist(confidences,bins='auto',rwidth=0.8)
        axs.set_xlabel('Tangle Confidences')
        axs.set_ylabel('Counts')

        # TODO: add F1 scores
        # make sure recall is sorted low to high
        idx = np.argsort(recall)
        
        # sorted arrays by idx
        confidences, precision, recall = confidences[idx], precision[idx], recall[idx]
        
        # print(recall)
        # first index where recall is above 0.80
        min_idx = np.argmax(recall >= 0.80)
        
        # calculate f1, only consider at above 0.80 recall
        f1 = 2*(precision*recall) / (precision + recall)
        optimal_idx = np.argmax(f1[min_idx:]) + min_idx
        optimal_confidence_threshold = confidences[optimal_idx]
        optimal_f1 = f1[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
        with open(os.path.join(args.odir,'%s_F1_precision_recall_%.2fIoU.txt' %(sana_io.get_antibody(ref_annos[0].file_name.replace('.json','.svs')),iou_threshold)),'w') as fp:
            fp.write('IoU: %0.2f\n' %iou_threshold)
            fp.write('optimal_confidence_threshold: %0.2f\n' %optimal_confidence_threshold)
            fp.write('optimal_f1: %0.2f\n' %optimal_f1)
            fp.write('optimal_precision: %0.2f\n' %optimal_precision)
            fp.write('optimal_recall: %0.2f\n' %optimal_recall)
        # plt.show()

        fname = args.odir.split('/')[-2]
        np.save(os.path.join(args.odir,'%s_%0.2fIoU_precision' %(fname,iou_threshold)),precision)
        np.save(os.path.join(args.odir,'%s_%0.2fIoU_recall' %(fname,iou_threshold)),recall)
        np.save(os.path.join(args.odir,'%s_%0.2fIoU_auc' %(fname,iou_threshold)),auc)

        # if prev_results <= len(results):
        #     print('prev_results: %d --> len(results): %d' %(prev_results,len(results)))
        #     prev_results = len(results)
        # if prev_seen <= len(seen):
        #     print('prev_seen: %d --> len(seen): %d' %(prev_seen,len(seen)))
        #     prev_seen = len(seen)
        # print('Precision:',precision)
        # print('Recall:',recall)
        print('IOU: %s...done' %iou_threshold)
        # plot the curve
        pr_dict['precision'].append(precision)
        pr_dict['recall'].append(recall)
        pr_dict['auc'].append(auc)
        
        # ax.plot(recall, precision,
        #         label='IOU = %.2f -- AuC = %.2f' % (iou_threshold, auc))
    #
    # end of iou_threshold loop

    # TODO: add F1 scores
    # make sure recall is sorted low to high
    # idx = np.argsort(recall)
    # thresholds, precision, recall = thresholds[idx], precision[idx], recall[idx]

    # # first index where recall is above 0.80
    # min_idx = np.argmax(recall >= 0.80)
    
    # # calculate f1, only consider at above 0.80 recall
    # f1 = 2*precision*recall / (precision + recall)
    # optimal_idx = np.argmax(f1[min_idx:]) + min_idx
    # optimal_confidence_threshold = thresholds[optimal_idx]
    # optimal_f1 = f1[optimal_idx]
    # optimal_precision = precision[optimal_idx]
    # optimal_recall = recall[optimal_idx]

    # finally, show the plot
    fig, ax = plt.subplots(1,1)
    for i, iouthresh in enumerate(args.iouthresh):
        ax.plot(pr_dict['recall'][i], pr_dict['precision'][i],
                label='IOU = %.2f -- AuC = %.2f' % (iouthresh, pr_dict['auc'][i]))
    ax.set_title(args.title+'\nTile ROI Count: %d | Ref Count: %d | Hyp Count: %d' %(len(roi_annos),len(ref_annos),len(hyp_annos)),fontsize=12)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.legend(loc='lower right')
    if not os.path.exists(os.path.join(args.odir,'IoUs')):
        os.makedirs(os.path.join(args.odir,'IoUs'))
    plt.savefig(os.path.join(args.odir, 'IoUs','pr_curves.png'), dpi=300)
    # plt.show()
    print('PNG saved:',args.odir)
# 
# end of get_iou_pr


def get_subregion_pr(args,logger=None):
    print('Generating PR curve for subregions: %s' %args.roisubregion)
    iou_threshold = args.iouthresh[0]
    
    pr_dict = {
        'precision': [],
        'recall': [],
        'auc': []
    }
    # loop through all provided roi subregions
    n_ref = []
    n_hyp = []
    prev_results = 0
    prev_seen = 0
    for i, roisubregion in enumerate(args.roisubregion):
        # filter ref and hyp annos to current subregion
        print('Loading subregion annos: %s' %roisubregion)
        ref_annos, hyp_annos = get_annos(args,load_subregion=roisubregion)

        # analyze the errors 
        if args.analyze_errors:
            error_analyze(ref_annos, hyp_annos, iou_threshold, args, load_subregion=roisubregion)

        # generate precision/recall curve, calculate the average precision
        precision, recall, auc, results, seen = pr(ref_annos, hyp_annos, iou_threshold, args)
        
        # add generate histogram check
        ref_sizes, hyp_sizes = get_poly_size(ref_annos,hyp_annos)
        nbins = 'auto'

        fig, ax = plt.subplots(1,1)
        n, bins, patches = ax.hist(ref_sizes,bins=nbins,alpha=0.5,label='Reference Anno\'s',color='blue',edgecolor='black')
        prob_density = stats.gaussian_kde(ref_sizes)
        ax.plot(bins,prob_density(bins),color='blue')

        n, bins, patches = ax.hist(hyp_sizes,bins=bins,alpha=0.5,label='Hypothesis Anno\'s',color='green',edgecolor='black')
        prob_density = stats.gaussian_kde(hyp_sizes)
        ax.plot(bins,prob_density(bins),color='green')
    
        ax.set_xlabel('Tangle Size (microns)')
        ax.set_ylabel('Tangle Count')
        fig.suptitle('Distribution of Tangle Area in %s' %str(roisubregion))
        plt.legend()
        if not os.path.exists(os.path.join(args.odir,'Subregions','Histograms')):
            os.makedirs(os.path.join(args.odir,'Subregions','Histograms'))
        plt.savefig(os.path.join(args.odir,'Subregions','Histograms','%s_histogram.png' %roisubregion), dpi=300)
    
        # print('Precision:',precision)
        # print('Recall:',recall)
        print('Subregion: %s...done' %roisubregion)
        pr_dict['precision'].append(precision)
        pr_dict['recall'].append(recall)
        pr_dict['auc'].append(auc)
        # plot the curve
        # ax.plot(recall, precision,
        #         label='Subregion = %s -- AuC = %.2f' % (roisubregion, auc))
        n_hyp.append(len(results))
        n_ref.append(len(seen))
    #
    # end of roi subregion loop

    # finally, show the plot
    fig, ax = plt.subplots(1,1)
    for i, roisubregion in enumerate(args.roisubregion):
        ax.plot(pr_dict['recall'][i], pr_dict['precision'][i],
                label='Subregion = %s | AuC = %.2f | Ref/Hyp Count: %d/%d' % (roisubregion, pr_dict['auc'][i],n_ref[i],n_hyp[i]))
    pr_title = args.title+' | IoU: %.2f\nTotal Ref Count: %d | Total Hyp Count: %d' %(iou_threshold,sum(n_ref),sum(n_hyp))
    ax.set_title(pr_title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.legend(loc='lower right')
    if not os.path.exists(os.path.join(args.odir,'Subregions')):
        os.makedirs(os.path.join(args.odir,'Subregions'))
    plt.savefig(os.path.join(args.odir,'Subregions','pr_curves.png'), dpi=300)
    print('PNG saved:',args.odir)
# 
# end of get_subregion_pr

# this function takes in any subset of ref/hyp annotations and returns a histogram
# of the distributions
def get_poly_size(ref_annos,hyp_annos):
    print('Calculating polygon areas...')
    ref_sizes = None
    hyp_sizes = None
    # all FN's or all FP's if either list is empty
    if len(hyp_annos)==0 or len(ref_annos)==0:
        return ref_sizes, hyp_sizes, ref_annos, hyp_annos

    ref_sizes = [ref_poly.area()*(0.5045**2) for ref_poly in ref_annos]
    hyp_sizes = [hyp_poly.area()*(0.5045**2) for hyp_poly in hyp_annos]

    ref_sizes = sorted(ref_sizes)
    hyp_sizes = sorted(hyp_sizes)

    return ref_sizes, hyp_sizes
#
# end of get_poly_size

def get_poly_ecc(ref_annos, hyp_annos):
    print('Calculating polygon eccentricity...')
    ref_eccs = None
    hyp_eccs = None
    if len(hyp_annos)==0 or len(ref_annos)==0:
        return ref_eccs, hyp_eccs, ref_annos, hyp_annos

    ref_eccs = [ref_poly.eccentricity() for ref_poly in ref_annos]
    hyp_eccs = [hyp_poly.eccentricity() for hyp_poly in hyp_annos]

    ref_eccs = sorted(ref_eccs)
    hyp_eccs = sorted(hyp_eccs)
    
    return ref_eccs, hyp_eccs



# this script takes a group of ref and hyp annotations, and outputs 1 or more
# PR curves based on the given cmdl arguments
def main(argv):

    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-slist', type=str, default='/Volumes/RESEARCH/BCHE300/all.list',help=SLIST_HELP)
    parser.add_argument('-lvl', type=int, default=0,help=LVL_HELP)
    parser.add_argument('-refdir', type=str, required=True, help=REFDIR_HELP)
    parser.add_argument('-sub_refdir', type=str, default=None, help=REFDIR_HELP)
    parser.add_argument('-hypdir', type=str, required=True, help=HYPDIR_HELP)
    parser.add_argument('-refclass', type=str, nargs='*', help=REFCLASS_HELP)
    parser.add_argument('-refname', type=str, nargs='*', help=REFCLASS_HELP)
    parser.add_argument('-hypclass', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-hypname', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-roiclass', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-roiname', type=str, default=None, help=ROICLASS_HELP)
    parser.add_argument('-roisubregion', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-iouthresh', type=float, nargs='*', default=[0.5],
                        help=IOUTHRESH_HELP)
    parser.add_argument('-softmax_thresh', type=float, default=0.25)
    parser.add_argument('-analyze_errors', action='store_true', default=False,
        help="toggle to analyze ROIs and examine TPs and FPs (still warns the user and expects manual input)")
    parser.add_argument('-odir', type=str, default='.')
    parser.add_argument('-title', type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    
    if args.iouthresh:
        get_iou_pr(args)

    # get_dice_hist(args)
    

    # if args.roisubregion:
    #     get_subregion_pr(args)

    # ref_annos, hyp_annos = get_annos(args)
    # if len(hyp_annos)>0:
    #     print('%d annotations loaded...' %len(hyp_annos))

    # ref_sizes, hyp_sizes = get_poly_size(ref_annos, hyp_annos)
    # nbins = 'auto'
    # fig, ax = plt.subplots(1,1)
    # fig.suptitle('Total Tangle Count vs Tangle Size')
    # # n, bins, patches = ax.hist(ref_sizes,bins=nbins,alpha=0.5,label='Reference Anno\'s',color='blue',edgecolor='black')
    # # prob_density = stats.gaussian_kde(ref_sizes)
    # # ax.plot(bins,prob_density(bins),color='blue')

    # n, bins, patches = ax.hist(hyp_sizes,bins=nbins,alpha=0.5,label='Hypothesis Anno\'s',color='green',edgecolor='black')
    # # prob_density = stats.gaussian_kde(hyp_sizes)
    # # ax.plot(bins,prob_density(bins),color='green')
    # plt.xticks(bins,rotation=60)
    # ax.set_xlabel('Tangle Size (microns$^2$)')
    # ax.set_ylabel('Tangle Count')
    # fig.suptitle('Distribution of Tangle Area')
    # plt.legend()
    # plt.tight_layout()
    # if not os.path.exists(os.path.join(args.odir,'IoUs','Histograms')):
    #     os.makedirs(os.path.join(args.odir,'IoUs','Histograms'))
    # plt.savefig(os.path.join(args.odir,'IoUs','Histograms','tangle_area.png'), dpi=300)
    # plt.show()

    # exit()
    print('sana_pr_curve...done!')
    
#
# end of main

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
