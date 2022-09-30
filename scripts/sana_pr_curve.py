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

# custom packages
from sana_io import get_anno_files, read_annotations, read_list_file, create_filepath
import sana_io
from sana_loader import Loader
from sana_geo import plot_poly
from sana_frame import Frame
from sana_geo import Converter
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
    x = x.copy().to_shapely()
    y = y.copy().to_shapely()
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

def get_annos(args):
   
    # get all annotation files in the ref and hyp dirs
    ref_files = sorted(get_anno_files(args.refdir))
    hyp_files = sorted(get_anno_files(args.hypdir+'detections/'))
    
    # debug loading
    # for hf in hyp_files:
    #     plot(args,hf)
    # exit()
    # loop through annotation files
    ref_annos, hyp_annos = [], []
    for rf in ref_files:
        for hf in hyp_files:
            # make sure there is a one to one match between ref and hyp dirs
            if os.path.basename(rf) == os.path.basename(hf):

                # load the ref rois, if given
                if args.roiclass is None:
                    rois = []
                else:
                    rois = []
                    for roi_class in args.roiclass:
                        rois += read_annotations(rf, class_name=roi_class, name=args.roiname)
                #
                # end of roi reading

                # load the ref annotations with any of the given ref classes
                if args.refclass is None:
                    ra = read_annotations(rf)
                else:
                    ra = []
                    for rclass in args.refclass:
                        ra += read_annotations(rf, class_name=rclass)
                #
                # end of ref anno reading

                # load the hyp annotations and confidences with a given hyp class
                if args.hypclass is None:
                    ha = read_annotations(hf)
                else:
                    ha = []
                    for hclass in args.hypclass:
                        ha += read_annotations(hf, class_name=hclass)
                #
                # end of hclass loop
                
                # keep only the annotations within a given roi
                # print('ref:')
                ref_annos += check_inside(rois, ra)
                # print('--ref_annos:',len(ref_annos))
                # print('hyp:')
                hyp_annos += check_inside(rois, ha)
                # print('--hyp_annos:',len(hyp_annos))
                # print()

                # ref_annos += ra
                # hyp_annos += ha              
            #
            # end of file matching check
        #
        # end of hyp file loop
    #
    # end of ref file loop

    # sort the hyp annotations from most to least confident
    hyp_annos.sort(key=lambda x: x.confidence, reverse=True)

    return ref_annos, hyp_annos
#
# end of get_annos

def plot(args, slide):
    # print(slide)
    # find the annotation file based on the slide filename
  
    # # access slides
    slides = sana_io.read_list_file(args.slist)

    img_dir = [s for s in slides if os.path.basename(s) == os.path.basename(slide).replace('.json','.svs')][0]

    # initialize loader, this will allow us to load slide data into memory
    image = sana_io.create_filepath(img_dir)
    
    loader = Loader(image)
    converter = loader.converter

    # tell the loader which pixel resolution to use
    loader.set_lvl(args.lvl)
    thumb_frame = loader.load_thumbnail()
    # find the annotation file based on the slide filename
    anno_fname = sana_io.create_filepath(slide, ext='.json', fpath=args.refdir)
    # print('slide:',slide)
    # print('anno_fname:',anno_fname)
  
    # load all ROIs with matching class name
    ROIS = []
    for roi_class in args.roiclass:
        ROIS += read_annotations(anno_fname, name=roi_class)
    # rescale the ROI annotations to the given pixel resolution
    for ROI in ROIS:
        if ROI.lvl != thumb_frame.lvl:
            converter.rescale(ROI, thumb_frame.lvl)

    # load all reference annotations with matching class name
    REF_ANNOS = []
    for ref_class in args.refclass:
        # load the annotations from the annotation file
        REF_ANNOS += read_annotations(anno_fname, name=ref_class)
    # rescale the ROI annotations to the given pixel resolution
    for REF_ANNO in REF_ANNOS:
        if REF_ANNO.lvl != thumb_frame.lvl:
            converter.rescale(REF_ANNO, thumb_frame.lvl)
    
    # load hypothesis annotations with matching class name
    HYP_ANNOS = []
    for hyp_class in args.hypclass:
        # load the annotations from the annotation file
        HYP_ANNOS += read_annotations(slide, class_name=hyp_class)
    # rescale the ROI annotations to the given pixel resolution
    for HYP_ANNO in HYP_ANNOS:
        if HYP_ANNO.lvl != thumb_frame.lvl:
            converter.rescale(HYP_ANNO, thumb_frame.lvl)

    # just use loader.thumbnail as the image you're plotting
    # plot the annotations
    for ROI in ROIS:
        # get the top left coordinate and the size of the bounding centroid
        loc, size = ROI.bounding_box()

        frame = loader.load_frame(loc, size)

        fig, axs = plt.subplots(1,2, sharex=False, sharey=False)

        # Plotting the processed frame and processed frame with clusters
        axs[0].imshow(thumb_frame.img, cmap='gray')
        for REF_ANNO in REF_ANNOS:
            # REF_ANNO.translate(loc)
            plot_poly(axs[0], REF_ANNO, color='red')

        axs[1].imshow(thumb_frame.img, cmap='gray')
        for HYP_ANNO in HYP_ANNOS:
            # HYP_ANNO.translate(loc)
            plot_poly(axs[1], HYP_ANNO, color='blue')
        plt.show()

# #
# end of pr


# TODO; add warning msg (DO NOT RUN ON BLIND DATASET, CONTINUE (Y/N)?)
def error_analyze(all_ref, all_hyp, iou_threshold, args):
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
        if args.roiclass is None:
            main_rois = []
        else:
            main_rois = []
            for roi_class in args.roiclass:
                main_rois += read_annotations(anno_fname, class_name=roi_class, name=args.roiname)
        #
        # end of roi reading
        
        # grab annos with same slide name
        slide_refs = [x for x in all_ref if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
        slide_hyps = [x for x in all_hyp if os.path.basename(x.file_name) == slide_name.replace('.svs','.json')]
        
        # loop through ROIs in a slide
        for main_roi_dir in main_roi_dirs:
            main_roi_name = os.path.split(main_roi_dir)[1]
            tile_class, tile_name = main_roi_name.split('_',1)
            # ex. tile_name = Tile 52
            
            # fetch main_roi based on class names and tile names (if exists)
            main_roi = [x for x in main_rois if (x.class_name == tile_class) and (x.name == tile_name)]
            if len(main_roi)==0:
                continue
            else:
                main_roi = main_roi[0]
            
            # ref, hyp = ...
            ref = [x for x in slide_refs if x.inside(main_roi)]  
            hyp = [x for x in slide_hyps if x.inside(main_roi)]  
            
            params_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','.csv'))
            params = Params(params_fname)

            
            # TODO: rename new_score to score
            # TODO: rename results/seen 
            results, seen, ref, hyp = score(ref, hyp, iou_threshold)
            
            # only do the following if there are errors!
            if (False in results) or (False in seen):
                conv = Converter()
                orig_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_ORIG.png'))
                orig_frame = Frame(orig_fname, lvl=0, converter=conv)
                padding = params.data['padding']

                # use sana_io.create_filepath function instead of this os.path.join
                wc_fname = os.path.join(main_roi_dir, slide_name.replace('.svs','_R13.npy'))
                wc_activation = np.load(wc_fname)
                wc_softmax = softmax(wc_activation,axis=0)

                lb_softmax = np.rint(255*wc_softmax[2,:,:]).astype(np.uint8)
                lb_softmax = cv2.resize(lb_softmax, orig_frame.size(),interpolation=cv2.INTER_LINEAR)
                lb_softmax = lb_softmax[:,:,None]
                
                # removing padding from original image and wildcat greyscale
                # orig_frame.img = orig_frame.img[padding//2:-padding//2,padding//2:-padding//2]
                # lb_softmax = lb_softmax[padding//2:-padding//2,padding//2:-padding//2]
                # [x.translate(padding//2+params.data['loc']) for x in ref]
                # [x.translate(padding//2+params.data['loc']) for x in hyp]

                [x.translate(params.data['loc']) for x in ref]
                [x.translate(params.data['loc']) for x in hyp]

                alpha = np.full_like(lb_softmax, 60)
                alpha[lb_softmax<np.floor(255*0.8)] = 0
                z = np.zeros_like(lb_softmax)

                rgba_wc = np.concatenate([z,lb_softmax,z,alpha],axis=2)

                                                   
                # print(len(results)==len(hyp))
                # print(len(seen)==len(ref))
                # print()

                fig, ax = plt.subplots(1,1)
                fig.suptitle('WildCat Probability of LB Detection >80%% \n %s | %s | IoU: %0.1f' %(slide_name.replace('.svs',''),tile_name,iou_threshold))
                
                ax.imshow(orig_frame.img)
                ax.imshow(rgba_wc)
                 
                # True in results is green, label=TP
                # False in results is red, label=FP
                plot_tp = True
                plot_fp = True
                for i, poly in enumerate(hyp):
                    if results[i] == True:
                        plot_poly(ax,poly,label='Hyp. TP' if plot_tp else '', color='green', linewidth=1.5)
                        plot_tp = False
                    if results[i] == False:
                        plot_poly(ax,poly,label='Hyp. FP' if plot_fp else '', color='red', linewidth=1.5)
                        plot_fp = False
                
                # True in seen is blue, label=hit
                # False in seen in orange, label=miss                     
                plot_hit = True
                plot_miss = True
                for i, poly in enumerate(ref):
                    if seen[i] == True:
                        plot_poly(ax,poly,label='Ref. Hit' if plot_hit else '', color='blue', linewidth=1.5)
                        plot_hit = False
                    if seen[i] == False:
                        plot_poly(ax,poly,label='Ref. Miss' if plot_miss else '', color='orange', linewidth=1.5)
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
                fig_fname = os.path.join(args.odir,'error_analysis',slide_name.replace('.svs','_IOU_%0.1f_ERRORS.png' %iou_threshold))
                plt.savefig(fig_fname)
                plt.close()
                # plt.show()
                
                # # debug WC activations for each of the classes
                # class_dict = {
                #     0: 'Artifact',
                #     1: 'Background',
                #     2: 'Lewy-Body',
                #     3: 'Lewy-Neurite'
                # }
                # fig, axs1 = plt.subplots(2,2)
                # axs1 = axs1.ravel()
                # for i in range(wc_softmax.shape[0]):
                #     axs1[i].imshow(wc_softmax[i,:,:],vmin=0,vmax=1)
                #     axs1[i].set_title(class_dict[i])
                # plt.show()

            
            


# NOTE: this can be any subset of a dataset! works on multiple slides and ROIs
# ERIC: error_analyze should loop through slides and ROI's in a slide
#        then extract the ref and hyp inside that ROI and pass to new_score
def score(ref_annos, hyp_annos, iou_threshold):
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
    parser.add_argument('-hypclass', type=str, nargs='*', help=HYPCLASS_HELP)
    parser.add_argument('-roiclass', type=str, nargs='*', help=ROICLASS_HELP)
    parser.add_argument('-roiname', type=str, default='', help=ROICLASS_HELP)
    parser.add_argument('-iouthresh', type=float, nargs='*', default=[0.5],
                        help=IOUTHRESH_HELP)
    parser.add_argument('-odir', type=str, default='.')
    parser.add_argument('-title', type=str, default="")
    args = parser.parse_args()


    # load the valid ref/hyp annos and the hyp confidences
    ref_annos, hyp_annos = get_annos(args)
    
    # print('...annotations loaded')
    # print('ref_annos:',len(ref_annos))
    # for ref in hyp_annos:
    #     print(ref.file_name)
    # # print('hyp_annos:',len(hyp_annos))
    # # for hyp in hyp_annos:
    # #     print(hyp)
    # exit()
    fig, ax = plt.subplots(1,1)

    # loop through all provided iou thresholds
    for iou_threshold in args.iouthresh:
        error_analyze(ref_annos,hyp_annos,iou_threshold,args)
        # generate precision/recall curve, calculate the average precision
        # precision, recall, auc = pr(ref_annos, hyp_annos, iou_threshold, args)
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
    plt.legend()
    plt.savefig(os.path.join(args.odir, 'pr_curves.png'), dpi=300)
    print('PNG saved:',args.odir)
#
# end of main

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
