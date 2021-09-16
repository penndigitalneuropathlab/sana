
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

    # filepath not given, use current filepath
    if fpath == "":
        fpath = ifpath

    # apply the replacement directory, if given
    elif rpath != "":
        fpath = os.path.dirname(ifile).replace(rpath, fpath)

    # construct the filepath
    ofile = get_fullpath(os.path.join(fpath, fname))

    # construct the new directory if needed
    create_directory(ofile)

    return ofile
#
# end of create_filepath

# loads a list of files into memory, checks to make sure each file exists
def read_list_file(list_f):

    # read the data from the filelist
    lines = [l.rstrip() for l in open(get_fullpath(list_f), 'r')]

    # keep only the files that actually exist
    return [l for l in lines if os.path.exists(l)]
#
# end of read_list_file

# loads the metrics stored by sana_gm_segmentation into memory
# TODO: need to do this in a different way... probably OOP
def read_metrics_file(f):

    # blank data if the file doesn't exist
    if not os.path.exists(f):
        return "", ["",""], ["",""], ["",""], "", ""

    # load the fields into memory
    fp = open(f, 'r')
    a, l0, l1, c0, c1, ds0, ds1, tt, st = \
        fp.read().split('\n')[0].split(',')

    # convert to proper datatypes
    angle = "" if a == "" else float(a)
    loc = ["",""] if l0 == "" else Point(float(l0), float(l1), False, 0)
    crop_loc = ["",""] if c0 == "" else Point(float(c0), float(c1), False, 0)
    ds = ["",""] if ds0 == "" else Point(float(ds0), float(ds1), False, 0)
    tissue_threshold = "" if tt == "" else float(tt)
    stain_threshold = "" if st == "" else float(st)

    return angle, loc, crop_loc, ds, tissue_threshold, stain_threshold
#
# end of read_metrics_file

# writes the metrics from sana_gm_segmentation to a file
# TODO: need to do this in a different way... probably OOP
def write_metrics_file(f, angle=None, loc=None, crop_loc=None, ds=None,
                       tissue_threshold=None, stain_threshold=None):

    # load the previous data if it already existed
    l, c, d = [["", ""]]*3
    a, tt, st = [""]*3
    if os.path.exists(f):
        a, l, c, d, tt, st = read_metrics_file(f)

    # if the metric was given as a parameter, store in the file
    if not angle is None: a = angle
    if not loc is None: l = loc
    if not crop_loc is None: c = crop_loc
    if not ds is None: d = ds
    if not tissue_threshold is None: tt = tissue_threshold
    if not stain_threshold is None: st = stain_threshold

    # write the data
    fp = open(f, 'w')
    fp.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
             (a, l[0], l[1], c[0], c[1], d[0], d[1], tt, st))
    fp.close()
#
# end of write_metrics_file

#<<<<<<< Updated upstream
#=======
# converts a Polygon annotation to a JSON, similar to GeoJSON
def anno_to_json(anno, class_name=None, anno_name=None, confidence=1.0):

    # generate a list of vertices from the Array
    verts = []
    for i in range(anno.shape[0]):
        verts.append([anno[i][0], anno[i][1]])

    # create the JSON format, using the given class and name
    annotation = {
        "type": "Feature",
        "id": "PathAnnotationObject",
        "geometry": {
          "type": "Polygon",
          "coordinates": [verts]
        },
        "properties": {
          "name": anno_name,
          "classification": {
            "name": class_name,
          },
          "confidence": confidence,
        }
    }
    return annotation
#
# end of anno_to_json

#>>>>>>> Stashed changes
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
    json.dump(json_annos, open(ofile, 'w'))
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

#
# end of file
