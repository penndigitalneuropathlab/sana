#!/usr/bin/env python3

# system modules
import os
import inspect
import sys
import math
import argparse
import time
from multiprocessing import Process

# installed modules
import numpy as np
from tqdm import tqdm

# custom modules
import sana_io
from sana_params import Params
from sana_loader import Loader
from sana_geo import transform_poly, transform_inv_poly, Point, Polygon
from sana_frame import Frame, create_mask
from sana_framer import Framer
from sana_logger import SANALogger

from sana_processors.NeuN_processor import NeuNProcessor
from sana_processors.SMI32_processor import SMI32Processor
from sana_processors.calretinin_processor import calretininProcessor
from sana_processors.MBP_processor import MBPProcessor
from sana_processors.SMI35_processor import SMI35Processor
from sana_processors.parvalbumin_processor import parvalbuminProcessor
from sana_processors.meguro_processor import MeguroProcessor
from sana_processors.AT8_processor import AT8Processor
from sana_processors.TDP43MP_processor import TDP43MPProcessor
from sana_processors.IBA1_processor import IBA1Processor
from sana_processors.R13_processor import R13Processor
from sana_processors.SYN303_processor import SYN303Processor
from sana_processors.HDAB_processor import HDABProcessor
from sana_processors.HE_processor import HEProcessor
from sana_processors.TauC3_processor import TauC3Processor

# debugging modules
from sana_geo import plot_poly
from matplotlib import pyplot as plt

# instantiates a Processor object based on the antibody of the svs slide
# TODO: where to put this
def get_processor(fname, frame, logger, **kwargs):
    try:
        antibody = sana_io.get_antibody(fname)
    except:
        antibody = ''
    antibody_map = {
        'NeuN': NeuNProcessor,
        'SMI32': SMI32Processor,
        'CALR6BC': calretininProcessor,
        'parvalbumin': parvalbuminProcessor,
        'SMI94': MBPProcessor,
        'SMI35': SMI35Processor,
        'MEGURO': MeguroProcessor,
        'AT8': AT8Processor,
        'TauC3': TauC3Processor,
        'C3': TauC3Processor,
        'MC1': AT8Processor,
        'TDP43MP': TDP43MPProcessor,
        'TDP43': TDP43MPProcessor,         # TODO: is this right?
        'IBA1': IBA1Processor,
        'R13': R13Processor,
        'MJFR13': R13Processor,
        'HE': HEProcessor,
        'SYN303': SYN303Processor,
        'aSYN': SYN303Processor,
        '': HDABProcessor,
    }
    cls = antibody_map[antibody]
    path = inspect.getfile(cls)
    proc = cls(fname, frame, logger, **kwargs)
    return proc

#
# end of get_processor

def get_slides(logger, lists, skip):
    slides = sana_io.get_slides_from_lists(lists)
    if len(slides) == 0:
        logger.error("No Slides Found")
    else:
        # skip a number of slides at the beginning of the list
        # TODO: make this divide by the number of jobs (skip should skip number slides AFTER splitting into jobs)
        # when I want to skip 5 slides, I want to skip the first 5 slides from ALL jobs
        slide_half_1 = slides[:len(slides)//2]
        slide_half_2 = slides[len(slides)//2:]
        slides = slide_half_1[skip:] + slide_half_2[skip:]
        logger.debug('Skipping %d slides from jobs!' % skip)
        
    return slides
#
# end of get_slides

def get_annotation_file(logger, slide_f, adir, rdir, ext='.json', overlap=None):
        
    # get the annotation file containing ROIs
    # anno_f = sana_io.create_filepath(
    #     slide_f, ext=ext, fpath=adir, rpath=rdir)

    # used for IBA1 microglia annotations
    anno_f = sana_io.create_filepath(
        slide_f.replace('-21_','-2021_'), ext=ext, fpath=adir, rpath=rdir)
    
    # make sure the file exists, else we skip this slide
    if not os.path.exists(anno_f):
        logger.warning('Annotation file %s does not exist\n' % anno_f)
        return None
    
    return anno_f
#
# end of get_annotation_file

def get_loader(logger, slide_f, lvl):
    try:
        loader = Loader(slide_f)
        loader.set_lvl(lvl)
        return loader
    except Exception as e:
        logger.warning('Could not load .svs file: %s' % e)
        return None
#
# end of get_loader

def load_rois(logger, anno_f, main_class, main_name):
    # logger.debug('Loading main_rois from %s' %os.path.basename(anno_f))
    main_rois = sana_io.read_annotations(anno_f, class_name=main_class, name=main_name)
    logger.debug('Number of main_rois found: %d' % len(main_rois))
    if len(main_rois) == 0:
        logger.debug('No ROIs found: %s' %os.path.basename(anno_f))
        # logger.debug('Input main_class/main_name: %s/%s' %(main_class,main_name))
        # logger.debug('No ROIs found for processing...possible Annotation Class Names are [%s]' % \
        #              ', '.join(set([x.class_name for x in sana_io.read_annotations(anno_f)])))
        # logger.debug('No ROIs found for processing...possible Annotation Main Names are [%s]' % \
        #              ', '.join(set([x.main_name for x in sana_io.read_annotations(anno_f)])))
    return main_rois
#
# end of load_main_rois

def process_slides(args, slides, logger):

    # loop through slides to load in all frames to process
    rois_to_process = []
    for slide_f in tqdm(slides):
        
        loader = get_loader(logger, slide_f, args.lvl)
        if loader is None:
            continue

        # find the locations of all the frames to process
        size = Point(args.frame_size, args.frame_size, is_micron=False, lvl=args.lvl)
        framer = Framer(loader, size)

        # generate the slide mask, if given
        # NOTE: this is generated from the main_rois and allows us to skip portions of the slide
        #       if they are not annotated. This will significantly speed up processing
        if args.use_mask:
            anno_f = get_annotation_file(logger, slide_f, args.adir, args.rdir)
            if anno_f is None:
                slide_mask = None
            else:
                rois = load_rois(logger, anno_f,
                                 args.main_class, args.main_name)
                if len(rois) == 0:
                    slide_mask = None
                else:
                    slide_mask = create_mask(rois, loader.thumbnail.size(),
                                             loader.thumbnail.lvl, loader.thumbnail.converter)
        else:
            slide_mask = None

        logger.debug('Looping through frames...')
        # loop through the frame locations
        for i, j in framer.inds():
            loc = framer.locs[i][j]

            # check if we want to actually process this frame            
            if slide_mask is None:
                do_process = False
            else:

                # grab the corresponding portion of the slide_mask
                loader.converter.rescale(loc, loader.thumbnail_lvl)
                loader.converter.rescale(size, loader.thumbnail_lvl)
                frame_mask = slide_mask.get_tile(loc, size)

                # only process if some of the mask is inside this frame
                do_process = np.sum(frame_mask != 0) > 0

            if do_process:

                # build the main ROI to process
                loader.converter.rescale(loc, args.lvl)
                loader.converter.rescale(size, args.lvl)
                x = [loc[0], loc[0]+size[0], loc[0]+size[0], loc[0]]
                y = [loc[1], loc[1], loc[1]+size[1], loc[1]+size[1]]
                main_roi = Polygon(x, y, is_micron=False, lvl=args.lvl)
                main_roi = main_roi.to_annotation(slide_f, sana_io.get_antibody(slide_f)+'_'+rois[0].class_name+'_ROI')                
                sub_rois = []
                roi_id = '%s_%d_%d' % (main_roi.class_name, i, j)

                # store the process arguments
                rois_to_process.append({
                    'args': args,
                    'slide': slide_f,
                    'main_roi': main_roi,
                    'main_roi_dict': main_roi.__dict__,
                    'sub_rois': sub_rois,
                    'sub_roi_dicts': [sub_roi.__dict__ for sub_roi in sub_rois],
                    'roi_id': roi_id,
                })
        #
        # end of frames loop
    #
    # end of slides loop

    # return the list of ROIs we extracted from the list of slides
    return rois_to_process
#
# end of process_slides

def process_rois(args, slides, logger, overlap_slides=None):
    
    print('Overlapping Slides:',overlap_slides)
    # loop through the slides
    rois_to_process = []
    for slide_f in tqdm(slides):
        logger.info('Setting up %s' % slide_f)
        
        if not args.sub_adir == '':
            # load in slides from respective anno folder

            # load in 
            if slide_f in overlap_slides:
                overlap_annos = {}
                for i, anno_fldr in enumerate(os.listdir(args.sub_adir)):
                    anno_f = get_annotation_file(logger, slide_f, os.path.join(args.sub_adir,anno_fldr), args.rdir)

                    if anno_f is None:
                        continue

                     # load the main roi(s) from the json file
                    if len(args.main_class) > 1:
                        slide_rois = []
                        for mclass in args.main_class:
                            slide_rois += sana_io.read_annotations(anno_f, class_name=mclass, name=args.main_name)
                    else:
                        slide_rois = load_rois(logger, anno_f, args.main_class[0], args.main_name)

                    overlap_annos[i] = slide_rois
                
                main_tiles = {}
                anno_keys = overlap_annos.keys()
                for k in anno_keys:
                    slide_annos = overlap_annos[k]
                    for anno in slide_annos:
                        if not anno.name in main_tiles.keys():
                            main_tiles[anno.name] = anno
                        elif anno.name in main_tiles.keys():
                            # have a 50% chance of overwriting anno already written in main_rois with that name
                            np.random.seed(1)
                            chance = np.random.randint(0, 10, 1)
                            prob = chance/10 
                            if prob >= 0.5:
                                main_tiles[anno.name] = anno
                [*main_names], [*main_rois] = zip(*main_tiles.items())

            else:
                for anno_fldr in os.listdir(args.sub_adir):
                    anno_f = get_annotation_file(logger, slide_f, os.path.join(args.sub_adir,anno_fldr), args.rdir)
                    if anno_f is None:
                        continue
                    else:
                        break

                # load the main roi(s) from the json file
                if len(args.main_class) > 1:
                    main_rois = []
                    for mclass in args.main_class:
                        main_rois += sana_io.read_annotations(anno_f, class_name=mclass, name=args.main_name)
                else:
                    main_rois = load_rois(logger, anno_f, args.main_class[0], args.main_name)
                    
                
        else:    
            anno_f = get_annotation_file(logger, slide_f, args.adir, args.rdir)
            
            if anno_f is None:
                continue
        
            # load the main roi(s) from the json file
            if len(args.main_class) > 1:
                main_rois = []
                for mclass in args.main_class:
                    main_rois += sana_io.read_annotations(anno_f, class_name=mclass, name=args.main_name)
            else:
                main_rois = load_rois(logger, anno_f, args.main_class[0], args.main_name)
        

        loader = get_loader(logger, slide_f, args.lvl)
        if loader is None:
            continue

        # loop through main roi(s)
        for main_roi_i, main_roi in enumerate(main_rois):

            # make sure the roi is properly formed
            if main_roi.shape[0] < 3:
                logger.warning('Skipping ROI... not a proper polygon')
                continue
            
            # load the sub roi(s) that are inside this main roi
            # NOTE: we attempt to find a sub_roi for each given sub_class
            # FIXME: fix to work to read in anno file from main_roi.file_name
            sub_rois = []
            for sub_class in args.sub_classes:
                rois = sana_io.read_annotations(anno_f, class_name=sub_class)
                for roi in rois:
                    if roi.partial_inside(main_roi) or main_roi.partial_inside(roi):
                        sub_rois.append(roi)
                        break
                # NOTE: None is used for not found sub rois since we still need to measure something
                else:
                    sub_rois.append(None)
                    logger.warning('Couldn\'t find the %s sub_roi' % sub_class)

            # create the ROI ID
            if not main_roi.name:
                roi_name = str(main_roi_i)
            else:
                roi_name = main_roi.name
            roi_id = '%s_%s' % (main_roi.class_name, roi_name)

            # store the process arguments for this ROI
            rois_to_process.append({
                'args': args,
                'slide': slide_f,
                'main_roi': main_roi,
                'main_roi_dict': main_roi.__dict__,
                'sub_rois': sub_rois,
                'sub_roi_dicts': [sub_roi.__dict__ if not sub_roi is None else None for sub_roi in sub_rois],
                'roi_id': roi_id,
            })
        #
        # end of main_rois loop
    #
    # end of slides loop

    # return a list of ROIs extracted from the list of slides
    return rois_to_process
#
# end of process_roi

def process(args, slide, first_run, roi_i, nrois, main_roi, main_roi_dict, sub_rois=[], sub_roi_dicts=[], roi_id=None):
    logger = SANALogger.get_sana_logger(args.debug_level)
    logger.info('Processing Frame (%d/%d)' % (roi_i, nrois))

    # reset the attributes on the annotations
    # NOTE: this is because pickling the Annotations loses the lvl attribute etc.
    # TODO: this is messy, we need to find a better way to pickle Annotation objects
    for x in main_roi_dict:
        main_roi.__setattr__(x, main_roi_dict[x])
    for i in range(len(sub_rois)):
        if not sub_roi_dicts[i] is None:
            for x in sub_roi_dicts[i]:
                sub_rois[i].__setattr__(x, sub_roi_dicts[i][x])

    loader = get_loader(logger, slide, args.lvl)
    if loader is None:
        return None
    
    # initialize the Params IO object, this will store parameters
    # relating to the loading/processing of the Frame, as well as
    # the various AO results
    params = Params()
    
    # create the directory for polygon detections
    detection_odir = sana_io.create_odir(args.odir, 'detections')
    
    # create the output directory path
    # NOTE: output/slide_name/slide_name.*
    slide_f = loader.fname
    slide_name = os.path.splitext(os.path.basename(slide_f))[0]
    odir = sana_io.create_odir(args.odir, slide_name)
    odir = sana_io.create_odir(odir, roi_id)
    logger.debug('Output directory successfully created: %s' % odir)

    # generate a quick plot of the thumbnail, and the main/sub ROIs we loaded in
    if logger.plots:
        logger.debug('Main ROI: %s' % str(main_roi))
        logger.debug('SHAPE: %s' % str(main_roi.shape))
        plot_rois = [main_roi.copy()] + [sub_roi.copy() for sub_roi in sub_rois]
        [loader.converter.rescale(x, loader.thumbnail.lvl) for x in plot_rois]
        colors = ['black', 'red']
                
        fig, ax = plt.subplots(1,1)
        ax.imshow(loader.thumbnail.img)
        [plot_poly(ax, x, color='red') for x in plot_rois[1:] if not x is None]
        plot_poly(ax, plot_rois[0], color='black')
        plt.show()
    
    # rescale the ROIs to the proper level
    loader.converter.rescale(main_roi, loader.lvl)
    [loader.converter.rescale(x, loader.lvl) for x in sub_rois if not x is None]
            
    # load the frame into memory using the main roi
    if args.mode == 'GM':
        # rotate/translate the coord. system to retrieve the frame from
        # the slide. the frame will be orthogonalized such that CSF is
        # at the top and WM is at the bottom of the image.
        frame = loader.load_gm_frame(logger, params, main_roi, padding=args.padding)
        
    elif args.mode == 'GMZONE':

        # same rotation goal as above, except we have to test all
        #  4 sides of the input ROI to find the CSF
        frame = loader.load_gm_zone_frame(logger, params, main_roi, padding=args.padding)
        
    else:
        # just translates the coord. system, no rotating or cropping
        frame = loader.load_roi_frame(logger, params, main_roi, padding=args.padding)

    # transform the ROIs to the Frame's coord. system
    transform_poly(
        main_roi,
        params.data['loc'], params.data['crop_loc'],
        params.data['M1'], params.data['M2']
    )
    for sub_roi_i in range(len(sub_rois)):
        if sub_rois[sub_roi_i] is None:
            continue
        transform_poly(
            main_roi,
            params.data['loc'], params.data['crop_loc'],
            params.data['M1'], params.data['M2']
        )

        # transform the sub ROIs to the Frame's coord. system
        for sub_roi_i in range(len(sub_rois)):
            if sub_rois[sub_roi_i] is None:
                continue
            transform_poly(
                sub_rois[sub_roi_i],
                params.data['loc'], params.data['crop_loc'],
                params.data['M1'], params.data['M2']
            )

    # get the processor object
    kwargs = {
        'qupath_threshold': args.qupath_threshold,
        'roi_type': args.mode,
        'save_images': args.save_images,
        'run_wildcat': args.run_wildcat,
    }

    # check for half_full debug level (skip ROI loading debugging plots and just generate processing debugging plots)
    if args.debug_level == 'half_full':
        logger.plots = True 

    processor = get_processor(slide_f, frame, logger, **kwargs)
    if processor is None:
        logger.warning('No processor found for %s' % slide_f)
        return None

    # finally, analyze the frame based on the antibody it was stained with
    processor.run(odir, detection_odir, first_run, params, main_roi, sub_rois)
#
# end of process

def process_job(pid, rois_to_process):
    n = len(rois_to_process)
    for i in range(n):
        roi = rois_to_process[i]
        roi['nrois'] = n
        roi['roi_i'] = i
        roi['first_run'] = i == 0
        process(**roi)
#
# end of process_job

def dispatch(rois_to_process, njobs):

    nrois = math.ceil(len(rois_to_process) / njobs)

    jobs = []
    for pid in range(njobs):
        st = pid * nrois
        en = (pid+1) * nrois
        
        # create, store, start the job
        p = Process(target=process_job, args=(pid, rois_to_process[st:en]))
        jobs.append(p)
        p.start()
    #
    # end of job dispatching

    return jobs
#
# end of dispatch

def main(argv):
    print(argv)
    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # setup the logger
    logger = SANALogger.get_sana_logger(args.debug_level)
    if logger.plots:
        logger.debug('Generating debugging plots...')  
    
    # get all the slide files to process
    slides = get_slides(logger, args.lists, args.skip)
    if len(slides) == 0:
        logger.error('No files found in lists: %s' % str(args.lists))
        exit()
    
    if not args.sub_adir == '':
        seen_slides = []
        overlap_slides = []
        anno_folders = os.listdir(args.sub_adir)
        for anno_fldr in anno_folders:
            anno_files = os.listdir(os.path.join(args.sub_adir,anno_fldr))            
            process_anno_files = [a for a in slides if os.path.basename(a).replace('.svs','.json') in anno_files]
            for a_file in process_anno_files:
                if a_file in seen_slides:
                    overlap_slides.append(a_file)
                else:
                    seen_slides.append(a_file)
        logger.debug('Overlapping slides: %d' %len(overlap_slides))
        
    # get the ROIs within the slides to process
    logger.info('Number of slides found: %d' % len(slides))
    if args.reprocess:    
        if args.mode == 'SLIDESCAN':

            # process the entire slide
            rois_to_process = process_slides(args, slides, logger)
        else:

            # process ROIs based on input annotations
            # if overlap_slides:
            #     rois_to_process = process_rois(args, slides, logger, overlap_slides=overlap_slides)
            # else:
            rois_to_process = process_rois(args, slides, logger)
            
        # create and start the jobs
        jobs = dispatch(rois_to_process, args.njobs)
            
        # join and wait until all jobs are finished        
        [job.join() for job in jobs]        
    logger.error('sana_process...done!')
    # TODO: put sana_results here
    
    # # TODO: put the SLIDESCAN aggregation here
    # for slide_f in slides:

    #     loader = get_loader(logger, slide_f, args.lvl)
    #     if loader is None:
    #         continue
    #     loader = get_loader(logger, slide_f, args.lvl)
    #     if loader is None:
    #         continue

    #     size = Point(args.frame_size, args.frame_size, is_micron=False, lvl=args.lvl)
    #     framer = Framer(loader, size)
    #     size = Point(args.frame_size, args.frame_size, is_micron=False, lvl=args.lvl)
    #     framer = Framer(loader, size)
        
    #     heatmap_measures = ['auto_ao', 'lb_wc_ao', 'lb_poly_ao']        
    #     heatmap = np.zeros((len(heatmap_measures), len(framer.locs)))
        
    #     d = sana_io.get_slide_odir(args.odir, slide_f)
    #     roi_dirs = [x for x in os.listdir(d) if os.path.isdir(x)]
    #     for roi_dir in roi_dirs:
    #         x, y = map(int, roi_dir.split('_')[:-2])

    #         params_f = os.path.join(roi_dir, os.path.basename(slide_f).replace('.svs', '.csv'))
    #         params = Params(params_f)
    #         for measure_i, measure in enumerate(heatmap_measures):
    #             heatmap[measure_i, x, y] = params.data[measure]
    #         params_f = os.path.join(roi_dir, os.path.basename(slide_f).replace('.svs', '.csv'))
    #         params = Params(params_f)
    #         for measure_i, measure in enumerate(heatmap_measures):
    #             heatmap[measure_i, x, y] = params.data[measure]

    # fig, axs = plt.subplots(2, heatmap.shape[0])
    # for i in range(heatmap.shape[0]):
    #     axs[0,i].imshow(heatmap[i])
    #     axs[1,i].imshow(interp_heatmap[i])
    # fig, axs = plt.subplots(2, heatmap.shape[0])
    # for i in range(heatmap.shape[0]):
    #     axs[0,i].imshow(heatmap[i])
    #     axs[1,i].imshow(interp_heatmap[i])
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-lists', type=str, nargs='*', required=True,
        help="filelists containing .svs files")
    parser.add_argument(
        '-mode', type=str, required=True, choices=['SLIDESCAN', 'ROI', 'GM', 'GMZONE'],
        help="method of how to process the slide, SLIDESCAN is WSI analysis, GM/GMZONE orthogonalize the ROI")
    parser.add_argument(
        '-njobs', type=int, default=1,
        help="number of jobs to deploy, NOTE: be careful with memory usage")
    parser.add_argument(
        '-reprocess', action='store_true', default=False,
        help="whether or not to actually run the processor.run() code")
    parser.add_argument(
        '-save_images', action='store_true', default=False,
        help="writes out intermediate image results to file -- WARNING: this will take up a lot of disk space with many slides")
    parser.add_argument(
        '-run_wildcat', action='store_true', default=False,
        help="runs the wildcat model (if available)")
    parser.add_argument(
        '-adir', type=str, default="",
        help="directory path containing .json files")
    parser.add_argument(
        '-sub_adir', type=str, default="",
        help="subdirectory path containing folders of .json files")
    parser.add_argument(
        '-odir', type=str, default="",
        help="directory path to write the results to")
    parser.add_argument(
        '-rdir', type=str, default="",
        help="directory path to replace")
    parser.add_argument(
        '-lvl', type=int, default=0,
        help="resolution level to use during processing")
    parser.add_argument(
        '-frame_size', type=int, default=1024,
        help="size of frame to load during SLIDESCAN")
    parser.add_argument(        
        '-use_mask', action='store_true', default=False,
        help="for SLIDESCAN, uses the main ROIs to reduce the number of frames to process the slide")
    parser.add_argument(
        '-main_class', type=str, nargs='*', default=None,
        help="ROI class used to load and process the Frame")
    parser.add_argument(
        '-main_name', type=str, default=None,
        help="name of annotation to match")
    parser.add_argument(
        '-sub_classes', type=str, nargs='*', default=[],
        help="class names of ROIs inside the main ROI to separately process")
    parser.add_argument(
        '-debug_level', type=str, default='normal',
        help="Logging debug level", choices=['full', 'half_full', 'debug', 'normal', 'quiet'])
    parser.add_argument(
        '-padding', type=int, default=0,
        help="Thickness of border to add to Frame to provide context for models")
    parser.add_argument(
        '-qupath_threshold', type=float, default=None,
        help="Pre-defined threshold in QuPath to use in Manual %AO"
    )
    parser.add_argument(
        '-stain_vector', type=float, nargs=6, default=None,
        help="Pre-defined stain vector in QuPath to use in Manual %AO"
    )
    parser.add_argument(
        '-skip', type=int, default=0
    )

    return parser
#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)

#
# end of file
