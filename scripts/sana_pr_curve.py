#!/usr/bin/env python

# system packages
import os
import sys
import argparse

# installed packages
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
import cv2
from shapely.geometry.multipolygon import MultiPolygon
from tqdm import tqdm

# custom packages
import sana_io
from sana_io import get_anno_files, read_annotations, read_list_file, create_filepath
from sana_loader import Loader
from sana_geo import Converter, plot_poly, Polygon, fix_polygon, from_shapely
from sana_frame import Frame
from sana_params import Params

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
                if annos[i].filter(roi).shape == annos[i].shape:
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

def get_annos(args,load_subregion=None):
   
    # get all annotation files in the ref and hyp dirs
    ref_files = sorted(get_anno_files(args.refdir))
    hyp_files = sorted(get_anno_files(os.path.join(args.hypdir,'detections')))
    
    if not len(ref_files):
        print('No ref files found...')
        exit()
    if not len(hyp_files):
        print('No hyp files found...')
        exit()

    # loop through annotation files
    ref_annos, hyp_annos = [], []
    for rf in ref_files:
        for hf in hyp_files:
            # make sure there is a one to one match between ref and hyp dirs
            if os.path.basename(rf) == os.path.basename(hf):

                # load the ref rois, if given
                rois = []
                if not args.roiname is None:
                    if not args.roiclass is None:
                        for roi_class in args.roiclass:
                            rois += read_annotations(rf, class_name=roi_class, name=args.roiname)
                    elif not load_subregion is None:
                        rois += read_annotations(rf, name=args.roiname, class_name=load_subregion)
                    else:
                        # print(args.roiname)
                        rois += read_annotations(rf, name=args.roiname)
                #
                # end of roi reading
                # print('len(loaded rois):',len(rois))

                if len(rois) == 0:
                    print(rf)
                    print('Could not read ROI annos...')

                # load the ref annotations with any of the given ref classes
                if args.refclass is None:
                    ra = read_annotations(rf)
                else:
                    ra = []
                    for rclass in args.refclass:
                        # print(rclass)
                        ra += read_annotations(rf, class_name=rclass)
                #
                # end of ref anno reading
                # print('len(loaded ref):',len(ra))

                
                if len(ra) == 0:
                    print('Could not read REF annos...')
                

                # load the hyp annotations and confidences with a given hyp class
                if args.hypclass is None:
                    ha = read_annotations(hf)
                else:
                    ha = []
                    for hclass in args.hypclass:
                        # print(hclass)
                        ha += read_annotations(hf, class_name=hclass)
                #
                # end of hclass loop
                # print('len(loaded hyp):',len(ha))

                if len(ha) == 0:
                    print('Could not read HYP annos...')
                
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
                # ref_annos += ra
                # hyp_annos += ha              
            #
            # end of file matching check
        #
        # end of hyp file loop
    #
    # end of ref file loop

    # sort the hyp annotations from most to least confident
    # hyp_annos = hyp_annos[::2]
    hyp_annos.sort(key=lambda x: x.confidence, reverse=True)

    if not len(ref_annos):
        print('Could not load ref annos...')
        exit()
    if not len(hyp_annos):
        print('Could not load hyp annos...')
        exit()

    print('len(ref_annos):',len(ref_annos))
    print('len(hyp_annos):',len(hyp_annos))

    return ref_annos, hyp_annos
#
# end of get_annos

def error_analyze(all_ref, all_hyp, iou_threshold, args, load_subregion=None):
    warning = input('DO NOT RUN ON BLIND DATASET, ARE YOU SURE YOU WANT TO CONTINUE? (Y/N)')
    if warning.lower() in ('yes','y'):
        img_dir = sana_io.read_list_file(args.slist)
        all_slides = [os.path.basename(x) for x in img_dir]
    
        for slide_name in all_slides:
            # slide_name = 2012-116-30F_R_MFC_R13_25K_05-23-19_DC.svs

            # build path name to get _Tile *'s
            # TODO: move this to a sana_io.get_output_file(slide_name,odir,suffix='_ORIG',extension='.json')
            bid = sana_io.get_bid(slide_name)
            region = sana_io.get_region(slide_name)
            antibody = sana_io.get_antibody(slide_name)
            odir = os.path.join(args.hypdir, bid, antibody, region)
            main_roi_dirs = [ f.path for f in os.scandir(odir) if f.is_dir() ]
            

            anno_fname = args.refdir+slide_name.replace('.svs','.json')
            # load the ref rois, if given
            main_rois = []
            if not args.roiname is None:
                if not args.roiclass is None:
                    for roi_class in args.roiclass:
                        main_rois += read_annotations(anno_fname, class_name=roi_class, name=args.roiname)
                elif not load_subregion is None:
                    main_rois += read_annotations(anno_fname, name=args.roiname, class_name=load_subregion)
                else:
                    # print(args.roiname)
                    main_rois += read_annotations(anno_fname, name=args.roiname)
            #
            # end of roi reading

            if len(main_rois) == 0:
                print('No main rois loaded --',anno_fname)
                continue

            # grab annos with same slide name
            slide_refs = [x for x in all_ref if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
            slide_hyps = [x for x in all_hyp if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
            # print('len(slide_refs):',len(slide_refs))
            # print('len(slide_hyps):',len(slide_hyps))

            # loop through ROIs in a slide
            for i, main_roi_dir in enumerate(main_roi_dirs):
                main_roi_name = os.path.split(main_roi_dir)[1]
                tile_class, tile_name = main_roi_name.split('_',1)
                # print(tile_class, '|' ,tile_name)
                
                # ex. tile_name = Tile 52
                
                # fetch main_roi based on class names and tile names (if exists)
                main_roi = [x for x in main_rois if (x.class_name == tile_class) and (x.name == tile_name)]
                
                # filter by subregion if doing subregion analysis
                if not load_subregion is None:
                    main_roi = [x for x in main_roi if x.class_name == load_subregion]

                if len(main_roi)==0:
                    # print('No main rois...')
                    continue
                else:
                    main_roi = main_roi[0]
                
                # ref, hyp = ...
                ref = [x for x in slide_refs if x.inside(main_roi)]  
                hyp = [x for x in slide_hyps if x.inside(main_roi)]  
                print('inside refs:',len(ref))
                print('inside hyps:',len(hyp))
                
                params_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','.csv'))
                params = Params(params_fname)
                
                # TODO: rename results/seen 
                results, seen, ref, hyp = score(ref, hyp, iou_threshold)
                
                # only do the following if there are errors!
                if (False in results) or (False in seen):
                    conv = Converter()
                    orig_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_ORIG.png'))
                    orig_frame = Frame(orig_fname, lvl=0, converter=conv)
                    padding = params.data['padding']

                    # use sana_io.create_filepath function instead of this os.path.join
                    wc_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_PROBS.npy'))
                    wc_activation = np.load(wc_fname)
                    wc_softmax = softmax(wc_activation,axis=0)
                    
                    # # LB detections are at dim 1 in wc_activations (and nonsoftmax'd)
                    # lb_activation = np.rint(255*wc_softmax[1,:,:]).astype(np.uint8)
                    # lb_activation = cv2.resize(lb_activation, orig_frame.size(),interpolation=cv2.INTER_LINEAR)
                    # lb_activation = lb_activation[:,:,None]

                    lb_softmax = np.rint(255*wc_softmax[2,:,:]).astype(np.uint8)
                    lb_softmax = cv2.resize(lb_softmax, orig_frame.size(),interpolation=cv2.INTER_LINEAR)
                    lb_softmax = lb_softmax[:,:,None]
                    
                    # # removing padding from original image and wildcat greyscale
                    # orig_frame.img = orig_frame.img[padding//2:-padding//2,padding//2:-padding//2]
                    # lb_softmax = lb_softmax[padding//2:-padding//2,padding//2:-padding//2]
                    # [x.translate(padding//2+params.data['loc']) for x in ref]
                    # [x.translate(padding//2+params.data['loc']) for x in hyp]


                    [x.translate(params.data['loc']) for x in ref]
                    [x.translate(params.data['loc']) for x in hyp]
                    roi_plot = main_roi.copy()
                    roi_plot.translate(params.data['loc'])

                    alpha = np.full_like(lb_softmax, 60)
                    # get probs > 30%
                    alpha[lb_softmax<255*0.3] = 0
                    z = np.zeros_like(lb_softmax)

                    rgba_wc = np.concatenate([z,lb_softmax,z,alpha],axis=2)
                                     
                    # print(len(results)==len(hyp))
                    # print(len(seen)==len(ref))
                    # print()


                    # Plot Hit/TPs and Miss/FPs overlay
                    fig, ax = plt.subplots(1,1)
                    fig.suptitle('WildCat Probability of Tangle Detection >30%% \n %s | %s-%s | IoU: %0.1f' %(slide_name.replace('.svs',''),tile_class,tile_name,iou_threshold))
                    
                    ax.imshow(orig_frame.img)
                    ax.imshow(rgba_wc)
                    plot_poly(ax,roi_plot,color='red')
                    
                    # True in results is green, label=TP
                    # False in results is red, label=FP
                    plot_tp = True
                    plot_fp = True
                    for i, poly in enumerate(hyp):
                        if results[i] == True:
                            plot_poly(ax,poly,label='Hyp. TP' if plot_tp else '', color='limegreen', linewidth=1.0)
                            plot_tp = False
                        if results[i] == False:
                            plot_poly(ax,poly,label='Hyp. FP' if plot_fp else '', color='yellow', linewidth=.75)
                            plot_fp = False
                    
                    # True in seen is blue, label=hit
                    # False in seen in orange, label=miss                     
                    plot_hit = True
                    plot_miss = True
                    for i, poly in enumerate(ref):
                        if seen[i] == True:
                            plot_poly(ax,poly,label='Ref. Hit' if plot_hit else '', color='blue', linewidth=.5)
                            plot_hit = False
                        if seen[i] == False:
                            plot_poly(ax,poly,label='Ref. Miss' if plot_miss else '', color='orange', linewidth=.75)
                            plot_miss = False
                    
                    # Shrink current axis by 20%
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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
                        fig_fname = os.path.join(args.odir,'IoUs','error_analysis',slide_name.replace('.svs','_IOU_%0.1f_ERRORS_%d.png' %(iou_threshold, i)))
                    
                    sana_io.create_directory(fig_fname)
                    plt.savefig(fig_fname)
                    plt.close()
                    # 
                    # end of TP/FP overlay plot
        print('Error analysis performed...')
    else:
        print('Error analysis NOT performed...')
                    # plt.show()
# 
# end of error_analyze
            
            


# NOTE: this can be any subset of a dataset! works on multiple slides and ROIs
def score(ref_annos, hyp_annos, iou_threshold):
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
    for hyp_i, hyp in enumerate(tqdm(hyp_annos)):

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
# end of score

def pr(ref_annos, hyp_annos, iou_threshold, args):
    print('Running Precision-Recall calculation...')

    # get the TP, FP, hits, misses, and sorted by confidence hyp annos
    results, seen, ref_annos, hyp_annos = score(ref_annos, hyp_annos, iou_threshold)

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

    return precision, recall, ap
#
# end of pr

def get_iou_pr(args):
    print('Generating PR curve for IoUs...')
    ref_annos, hyp_annos = get_annos(args)
    print('Annotations loaded...')
    fig, ax = plt.subplots(1,1)

    # loop through all provided iou thresholds
    for iou_threshold in args.iouthresh:
        error_analyze(ref_annos,hyp_annos,iou_threshold,args)
        
        # generate precision/recall curve, calculate the average precision
        precision, recall, auc = pr(ref_annos, hyp_annos, iou_threshold, args)
        
        # print('Precision:',precision)
        # print('Recall:',recall)
        print('IOU: %s...done' %iou_threshold)
        # plot the curve
        ax.plot(recall, precision,
                label='IOU = %.2f -- AuC = %.2f' % (iou_threshold, auc))
    #
    # end of iou_threshold loop

    # finally, show the plot
    ax.set_title(args.title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.legend(loc='lower right')
    if not os.path.exists(os.path.join(args.odir,'IoUs')):
        os.makedirs(os.path.join(args.odir,'IoUs'))
    plt.savefig(os.path.join(args.odir, 'IoUs','pr_curves.png'), dpi=300)
    print('PNG saved:',args.odir)
# 
# end of get_iou_pr


def get_subregion_pr(args,analyze=False):
    print('Generating PR curve for subregions: %s' %args.roisubregion)
    iou_threshold = args.iouthresh[0]
    fig, ax = plt.subplots(1,1)

    # loop through all provided iou thresholds
    for roisubregion in args.roisubregion:
        # filter ref and hyp annos to current subregion
        ref_annos, hyp_annos = get_annos(args,load_subregion=roisubregion)

        # analyze the errors 
        if analyze:
            error_analyze(ref_annos, hyp_annos, iou_threshold, args, load_subregion=roisubregion)

        # generate precision/recall curve, calculate the average precision
        precision, recall, auc = pr(ref_annos, hyp_annos, iou_threshold, args)
        
        # print('Precision:',precision)
        # print('Recall:',recall)
        print('Subregion: %s...done' %roisubregion)
        # plot the curve
        ax.plot(recall, precision,
                label='Subregion = %s -- AuC = %.2f' % (roisubregion, auc))
    #
    # end of iou_threshold loop

    # finally, show the plot
    pr_title = args.title+' | IoU: %.2f' %iou_threshold
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


# this script takes a group of ref and hyp annotations, and outputs 1 or more
# PR curves based on the given cmdl arguments
def main(argv):

    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-slist', type=str, default='/Volumes/RESEARCH/BCHE300/all.list',help=SLIST_HELP)
    parser.add_argument('-lvl', type=int, default=0,help=LVL_HELP)
    parser.add_argument('-refdir', type=str, required=True, help=REFDIR_HELP)
    parser.add_argument('-hypdir', type=str, required=True, help=HYPDIR_HELP)
    parser.add_argument('-refclass', type=str, nargs='*', help=REFCLASS_HELP)
    parser.add_argument('-refname', type=str, nargs='*', help=REFCLASS_HELP)
    parser.add_argument('-hypclass', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-hypname', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-roiclass', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-roiname', type=str, default='', help=ROICLASS_HELP)
    parser.add_argument('-roisubregion', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-iouthresh', type=float, nargs='*', default=[0.5],
                        help=IOUTHRESH_HELP)
    parser.add_argument('-odir', type=str, default='.')
    parser.add_argument('-title', type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    get_iou_pr(args)

    get_subregion_pr(args,analyze=True)
    # exit()
    
#
# end of main

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
