
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tifffile import TiffWriter
import numpy as np
import geojson
from tqdm import tqdm

from matplotlib import pyplot as plt

import pdnl_sana as sana
import pdnl_sana.geo
import pdnl_sana.slide

def dispatch_jobs(job: Callable, job_args: list[dict], n_cores: int=1, progress_str: str=""):
    """
    This function distributes jobs across multiple processes and yields the results as they complete
    :param job: callable function
    :param job_args: list of dictionaries containing keyword arguments to pass to the job
    :param n_cores: number of cores to use for multi-processing, default is 1
    :param progress_str: description for progress bar
    """
    # don't do any multiprocessing, just yield the results
    if n_cores == 1:
        for args in tqdm(job_args, desc=progress_str):
            yield job(**args)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit the jobs
            futures = {executor.submit(job, **args) for args in job_args}

            # yield the results as they appear
            for future in tqdm(as_completed(futures), total=len(futures), desc=progress_str):
                yield(future.result())

def read_geojson(f: str) -> list[pdnl_sana.geo.Polygon]:
    """
    This function converts polygonal arrays in .geojson format into pdnl_sana.geo.Polygons
    :param f: path to .geojson file
    """
    annotations = []
    data = geojson.load(open(f, 'r'))
    if 'features' in data:
        data = data['features']
    for ann in data:
        props = ann.get('properties', {})
        class_name = props.get('classification', {}).get('name', None)
        name = props.get('name', None)
        if ann['geometry']['type'] == 'MultiPolygon':
            
            for x in ann['geometry']['coordinates']:
                xy = np.squeeze(np.array(x)).astype(float)
                print(xy.shape)
                annotation = pdnl_sana.geo.Annotation(
                    *xy.T, 
                    class_name=class_name, annotation_name=name, 
                    level=0
                )
                annotations.append(annotation)
        else:
            xy = np.squeeze(np.array(ann['geometry']['coordinates'])).astype(float)
            annotation = pdnl_sana.geo.Annotation(
                *xy.T, 
                class_name=class_name, annotation_name=name, 
                level=0
            )

            # shape checking
            if (len(annotation.shape) != 2) or \
            (annotation.shape[0] < 2) or \
            (annotation.shape[1] != 2):
                print(f"ERROR: Improper polygon -- {annotation.shape} | {annotation.class_name}")
                exit()

        annotations.append(annotation)

    return annotations

def write_geojson(fpath: str, annotations):
    with open(fpath, 'w') as fp:
        geojson.dump([x.to_geojson() for x in annotations], fp)

def save_arrays(fpath: str, **kwargs):
    """
    Adds arrays to an outputs archive, keeping the ones already in it. A zip
    cannot be appended to in place, so the existing members are rewritten with
    the new ones, via a temporary file so a partial write can't destroy them.
    :param fpath: path to .npz file
    :param arrays:
    """
    saved = {}
    if os.path.exists(fpath):
        with np.load(fpath) as z:
            saved = dict(z)
    saved.update(kwargs)
    np.savez_compressed(fpath+'.tmp.npz', **saved)
    os.replace(fpath+'.tmp.npz', fpath)

def crop_wsi(input_path, output_path, loc, size, debug=False):
    logger = sana.logging.Logger('normal')
    loader = sana.slide.Loader(logger, input_path)
    tb = loader.load_thumbnail()

    if debug:    
        rect = sana.geo.rectangle_like(loc, loc, size)
        rect = loader.converter.to_pixels(rect, level=tb.level)
        fig, ax = plt.subplots(1,1)
        ax.imshow(tb.img)
        ax.plot(*rect.T, color='red')
        plt.show()

    cropped_wsi = loader.load_frame(loc, size, level=0)

    with TiffWriter(output_path, bigtiff=False) as tif:
        metadata = {
            # "tiff.XResolution": loader.mpp,
            # "tiff.YResolution": loader.mpp,
            # "tiff.ResolutionUnit": 'µm',
        }
        options = {
            "photometric": "rgb",
            "tile": (32,32),
            "compression": "jpeg",
            "metadata": metadata,
        }

        res = (1e4/loader.mpp, 1e4/loader.mpp)
        subresolutions = len(loader.level_downsamples)-1
        tif.write(
            cropped_wsi.img, 
            subifds=None, 
            resolution=res,
            resolutionunit="MICROMETER",
            dtype=np.uint8,
            shape=cropped_wsi.img.shape,
            **options,
        )
        for level in range(1, len(loader.level_downsamples)):
            mag = 4**level
            ds = loader.level_downsamples[level]
            resized = cropped_wsi.copy()
            new_size = cropped_wsi.size()/ds
            print(ds)
            print(new_size)
            resized.resize(new_size, interpolation=cv2.INTER_AREA)
            res = (1e4/mag/loader.mpp, 1e4/mag/loader.mpp)
            print(res)
            print(resized.img.shape)
            tif.write(
                resized.img, 
                subfiletype=1, 
                # resolution=res,
                dtype=np.uint8,
                shape=resized.img.shape,
                **options,
            )
        thumbnail_ds = loader.level_downsamples[-1]*2
        thumbnail = cropped_wsi.copy()
        thumbnail.resize(cropped_wsi.size()/ds, interpolation=cv2.INTER_LINEAR)
        # tif.write(thumbnail.img, metadata={'Name': 'thumbnail'})

    loader = sana.slide.Loader(logger, output_path)
    print(loader.mpp, loader.level_downsamples)
