
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import geojson
from tqdm import tqdm

import pdnl_sana.geo

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
        xy = np.squeeze(np.array(ann['geometry']['coordinates'])).astype(float)
        props = ann.get('properties', {})
        class_name = props.get('classification', {}).get('name', None)
        name = props.get('name', None)
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
