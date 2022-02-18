
# system packages
import os
import sys
import json

# installed packages
import numpy as np

# sana packages
from sana_geo import Polygon, Point, Annotation

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
def get_anno_files(d):
    f = get_files(d)
    return [file for file in f if is_anno(file)]
#
# end of get_anno_files

# returns True if Loader can handle the given file
def is_slide(f):
    try:
        Loader(f)
        return True
    except:
        return False
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
    ifname = os.path.basename(ifile)
    ifname, iext = os.path.splitext(ifname)

    # extension not given, use current extension
    if ext == "":
        ext = iext

    # create the output filename using the basename, suffix, and extension
    fname = '%s%s%s' % (ifname, suffix, ext)

    # create the path to the file
    fpath = get_fpath(ifpath, fpath, rpath)

    # construct the filepath
    ofile = get_fullpath(os.path.join(fpath, fname))

    return ofile
#
# end of create_filepath

# loads a list of files into memory, checks to make sure each file exists
def read_list_file(list_f):

    # read the data from the filelist
    lines = [get_fullpath(l.rstrip()) for l in open(get_fullpath(list_f), 'r')]

    # keep only the files that actually exist
    return [l for l in lines if os.path.exists(l)]
#
# end of read_list_file

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
def read_annotations(ifile, class_name=None):

    # remove unwanted header bytes if they exist
    fix_annotations(ifile)

    # blank data if the file doesn't exist
    if not os.path.exists(ifile):
        return []

    # load the json data
    fp = open(ifile, 'r')
    data = json.loads(fp.read())

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
        annotations = [a for a in annotations if a.class_name == class_name]

    return annotations
#
# end of read_annotations

# writes a list of Polygon annotations to a JSON annotation file
#  -ofile: location to write the annotations to
#  -annos: list of Polygon Annotations
#  -class_names: list of class names, blank if not given
#  -anno_names: list of anno names, blank if not given
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
#  -class_names: list of class names, blank if not given
#  -anno_names: list of anno names, blank if not given
def append_annotations(ofile, annos, class_names=None, anno_names=None):

    # provide blank names if not given
    if class_names is None:
        class_names = ['']*len(annos)
    if anno_names is None:
        anno_names = ['']*len(annos)

    # load the original annotations
    orig_annos, orig_cnames, orig_anames = read_annotations(ofile)

    # append the new annotations to the old annotations
    annos = orig_annos + annos
    class_names = orig_cnames + class_names
    anno_names = orig_anames + anno_names

    # write the data
    write_annotations(ofile, annos, class_names, anno_names)
#
# end of append_annotations

# this class reads and writes data associated with processed Frames
# NOTE: when adding fields, be sure to edit the following:
#           self.data
#           parse_val()
#           write_data()
class DataWriter:
    def __init__(self, fname):

        # initalize the data
        self.data = {
            'lvl': None,
            'loc': None,
            'size': None,
            'ao': None,
            'aos_list': [],
            'csf_threshold': None,
            'stain_threshold': None,
            'angle': None,
            'density': None,
            'crop_loc': None,
            'crop_size': None,
            'ds': None,
        }
        self.line = '%s\t%s\n'

        # load the data from the filename into memory
        self.read_data(fname)
    #
    # end of constructor

    def read_data(self, fname):
        self.fname = fname

        # make sure the file exists already
        if not os.path.exists(self.fname):
            return

        # load the fields into memory
        for line in open(self.fname, 'r'):
            key, val = line.split('\t', maxsplit=1)
            val = self.parse_val(key, val.rstrip())
            if not val is None:
                self.data[key] = val
        #
        # end of reading data
    #
    # end of read_data

    def write_data(self):
        fp = open(self.fname, 'w')
        fp.write(self.line % ('lvl', self.write_int(self.data['lvl'])))
        fp.write(self.line % ('loc', self.write_point(self.data['loc'])))
        fp.write(self.line % ('size', self.write_point(self.data['size'])))
        fp.write(self.line % ('ao', self.write_float(self.data['ao'])))
        fp.write(self.line % ('density', self.write_float(self.data['density'])))
        fp.write(self.line % ('aos_list',
                              self.write_float_list(self.data['aos_list'])))
        fp.write(self.line % ('csf_threshold',
                              self.write_int(self.data['csf_threshold'])))
        fp.write(self.line % ('stain_threshold',
                              self.write_int(self.data['stain_threshold'])))
        fp.write(self.line % ('angle',
                              self.write_float(self.data['angle'])))
        fp.write(self.line % ('crop_loc',
                              self.write_point(self.data['crop_loc'])))
        fp.write(self.line % ('crop_size',
                              self.write_point(self.data['crop_size'])))
        fp.write(self.line % ('ds',
                              self.write_point(self.data['ds'])))
        fp.close()
    #
    # end of write_data

    def parse_val(self, key, val):
        if len(val) == 0:
            return None
        elif key == 'lvl':
            return self.parse_int(val)
        elif key == 'loc':
            return self.parse_point(val)
        elif key == 'size':
            return self.parse_point(val)
        elif key == 'ao':
            return self.parse_float(val)
        elif key == 'aos_list':
            return self.parse_float_list(val)
        elif key == 'csf_threshold':
            return self.parse_int(val)
        elif key == 'stain_threshold':
            return self.parse_int(val)
        elif key == 'angle':
            return self.parse_float(val)
        elif key == 'crop_loc':
            return self.parse_point(val)
        elif key == 'crop_size':
            return self.parse_point(val)
<<<<<<< HEAD
=======
        elif key == 'ds':
            return self.parse_point(val)
>>>>>>> a8183d0f585d3bf28971d7166953986deb47c88f
        else:
            return None
    #
    # end of parse_val

    def parse_int(self, val):
        return int(val)
    def parse_float(self, val):
        return float(val)
    def parse_point(self, val):
        x0, x1 = [float(x) for x in val.split('\t')]
        return Point(x0, x1, False, 0)
    def parse_float_list(self, val):
        return [float(x) for x in val.split('\t') if len(x) != 0]
    #
    # end of parsing

    def write_int(self, val):
        return "%d" % (val) if not val is None else ""
    def write_float(self, val):
        return '%.6f' % (val) if not val is None else ""
    def write_point(self, val):
        return '%.6f\t%.6f' % (val[0], val[1]) if not val is None else ""
    def write_float_list(self, val):
        return '%.6f\t'*len(val) % tuple(val)
    #
    # end of writing
#
# end of DataWriter

#
# end of file
