
# system packages
import os
import ntpath
import sys
import json
import re
import fnmatch

# installed packages
import numpy as np
import geojson

# sana packages
from sana_geo import Polygon, Point, Annotation

class SlideNameException(Exception):
    def __init__(self, fname, message='ERROR: Cannot parse slide file: %s'):
        self.message = message % fname
        super().__init__(self.message)
    

# resolves relative filepaths and ~
#  e.g. ~/data/x.svs -> /Users/yourname/data/x.svs
#  e.g. ./data/x.svs -> /Users/yourname/data/x.svs
def get_fullpath(f):
    return os.path.abspath(os.path.expanduser(f))
#
# end of get_fullpath

# creates a new directory if it doesn't exist, else does nothing
def create_directory(f):
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))
#
# end of create_directory

# gets the fullpath to every file (recursively or not) in a given directory
# NOTE: returns nothing if directory does not exist
def get_files(d, recurse=False):

    # make sure the directory exists
    if not os.path.exists(d):
        return []

    f = []
    if recurse:
        for root, _, files in os.walk(d):
            f += [os.path.join(root, file) for file in files]
    else:
        f = [os.path.join(d, file) for file in os.listdir(d)]
    return f
#
# end of get_files

# gets all .svs slide files in a given directory
def get_slide_files(d):
    f = get_files(d)
    return [file for file in f if is_slide(file)]
#
# end of get_slide_files

# gets all .json annotation files in a given directory
def get_anno_files(d,recurse=False):
    f = get_files(d,recurse)
    return [file for file in f if is_anno(file)]
#
# end of get_anno_files

# returns True if Loader can handle the given file
def is_slide(f):
    return f.endswith('.svs')
#
# end of is_slide

# returns True if read_annotations can handle the given file
def is_anno(f):
    try:
        read_annotations(f)
        return True
    except Exception as e:
        return False
#
# end of is_anno

def get_slide_id(fname):
    return fname.split('_')[0]

def get_slide_name(fname):
    return os.path.splitext(os.path.basename(fname))[0]

def get_slide_parts(fname):
    if not is_slide(fname):
        raise SlideNameException(fname)

    fname = ntpath.basename(fname)
    parts = fname.split('_')
    if len(parts) == 7:
        cid, hemi, region, antibody, dil, date, init = parts
    elif len(parts) == 6:
        if parts[1] in ['L','R','N']:
            cid, hemi, region, antibody, date, init = parts
            dil = None
        else:
            cid, region, antibody, dil, date, init = parts
            hemi = 'N'
    else:
        raise Exception

    return cid, hemi, region, antibody, dil, date, init
#
# end of get_slide_parts

# e.g. 2017-191-25F
def get_bid(fname):
    return get_slide_parts(fname)[0]
# e.g. 2017-191
def get_aid(fname):
    return '-'.join(get_slide_parts(fname)[0].split('-')[:2])
# e.g. 25F
def get_bnum(fname):
    return get_slide_parts(fname)[0].split('-')[2]
# e.g. L, R, or N
def get_hemi(fname):
    return get_slide_parts(fname)[1]
# e.g. MFC
def get_region(fname):
    return get_slide_parts(fname)[2]
# e.g. SMI32
def get_antibody(fname):
    return get_slide_parts(fname)[3]

def get_fpath(ifpath, fpath="", rpath=""):

    # filepath not given, use current filepath
    if fpath == "":
        fpath = ifpath

    # apply the replacement directory, if given
    elif rpath != "":
        fpath = os.path.dirname(ifile).replace(rpath, fpath)

    return fpath
#
# end of get_fpath

# creates a new filepath given an existing file and various parameters
#  -ifile: input filename, the path, suffix, and extension will be modified
#  -ext: file extension to be used, extension not changed if not given
#  -fpath: filepath to be used, not changed if not given
#  -rpath: a portion of the path in ifile to be replaced with fpath
#  e.g. $fpath/$ifile$suffix$ext
#  e.g. ifile=./data1/data2/slides/slide.svs
#       ext=.json
#       suffix=_DETECTIONS
#       fpath=./output
#       rpath=./data1/data2
#       result=./output/slides/slide_DETECTIONS.json
def create_filepath(ifile, ext="", suffix="", fpath="", rpath=""):

    # get the parts of the original filepath
    ifpath = os.path.dirname(ifile)
    ifname = ntpath.basename(ifile)
    ifname, iext = os.path.splitext(ifname)

    # extension not given, use current extension
    if ext == "":
        ext = iext

    if suffix != "":
        suffix = '_' + suffix

    # create the output filename using the basename, suffix, and extension
    fname = '%s%s%s' % (ifname, suffix, ext)

    # create the path to the file
    fpath = get_fpath(ifpath, fpath, rpath)

    # construct the filepath
    #ofile = get_fullpath(os.path.join(fpath, fname))
    ofile = os.path.join(fpath, fname)

    return ofile
#
# end of create_filepath

def get_slide_odir(odir, slide):
    return os.path.join(odir, get_bid(slide), get_antibody(slide), get_region(slide))

def get_params_file(odir, slide_name):
    return os.path.join(odir, slide_name+'.csv')

# creates a subdirectory for the ith ROI in a given json file
def create_odir(odir, s):
    odir = '%s/%s' % (odir, s)
    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir
#
# end of create_odir

# loads a list of files into memory, checks to make sure each file exists
def read_list_file(list_f):

    # read the data from the filelist
    files = [get_fullpath(l.rstrip()) for l in open(get_fullpath(list_f), 'r')]
    return files
#
# end of read_list_file

# generates a list of slide files from a series of lists from the commandline
def get_slides_from_lists(lists):

    # concatenate all slide lists
    slides = []
    for f in lists:
        if os.path.exists(f):
            slides += read_list_file(f)
    return slides
#
# end of get_slides_from_lists

# removes unreadable header data from JSON annotation files
# NOTE: these headers come export JSON files from Qupath
def fix_annotations(ifile):

    # load the data as bytes
    fp = open(ifile, 'rb')
    data = fp.read()
    fp.close()

    # find the index of the first annotation in the json
    ind = data.find(b'[\n')
    if ind == -1:
        ind = data.find(b'[]')
        if ind == -1:
            return
    # rewrite the data starting at the first annotation
    fp = open(ifile, 'wb')
    fp.write(data[ind:])
    fp.close()
#
# end of fix_annotation

# loads a JSON annotation file into memory
#  -ifile: input JSON file to be read
#  -class_name: if given, only returns annotations with this class
def read_annotations(ifile, class_name=None, name=None):
    if ifile.endswith('.geojson'):
        data = geojson.load(open(ifile, 'r'))['features']
    elif ifile.endswith('.json'):
        # remove unwanted header bytes if they exist
        fix_annotations(ifile)

        # blank data if the file doesn't exist
        if not os.path.exists(ifile):
            return []

        # load the json data
        fp = open(ifile, 'r', encoding='utf-8')
        data = json.loads(fp.read())
    else:
        raise Exception

    # load the annotations
    # NOTE: this could be handled by a GeoJSON package?
    annotations = []
    for annotation in data:

        # get the xy coordinates from the geometry of the annotation
        geo = annotation['geometry']

        # get the class name, if exists
        if 'classification' not in annotation['properties']:
            cname = ""
        else:
            cname = annotation['properties']['classification']['name']
        #
        # end of class name reading

        # get the anno name, if exists
        if 'name' not in annotation['properties']:
            aname = ""
        else:
            aname = annotation['properties']['name']
        #
        # end of anno name reading

        # get the confidence, if exists
        if 'confidence' not in annotation['properties']:
            confidence = 1.0
        else:
            confidence = float(annotation['properties']['confidence'])
        #
        # end of confidence reading

        # create and store the annotation object
        annotations.append(
            Annotation(geo, ifile, cname, aname,
                       confidence=confidence, is_micron=False, lvl=0))
    #
    # end of annotation loop

    # only return annotations with the given class name
    if not class_name is None:
        annotations = [a for a in annotations if fnmatch.fnmatch(a.class_name, class_name)]
    if not name is None:
        annotations = [a for a in annotations if fnmatch.fnmatch(a.name, name)]

    return annotations
#
# end of read_annotations

# writes a list of Polygon annotations to a JSON annotation file
#  -ofile: location to write the annotations to
#  -annos: list of Polygon Annotations
def write_annotations(ofile, annos):

    # convert the Ann objects to json strings
    json_annos = [anno.to_json() for anno in annos]

    # write the file
    json.dump(json_annos, open(ofile, 'w'), indent=2)
#
# end of write_annotations

# appends a list of Polygon annotations to an existing JSON annotation file
#  -ofile: location of existing file to write to
#  -annos: list of Polygon annotations
def append_annotations(ofile, annos):
    # checks if annotation file exists
    if os.path.exists(ofile):
        orig_annos = read_annotations(ofile)

        # if file is not empty, add new annos to old annos
        if orig_annos:
            annos = orig_annos + annos
    else:
        print('Output directory does not exist:',ofile)
    # write the data
    write_annotations(ofile, annos)

#
# end of append_annotations
