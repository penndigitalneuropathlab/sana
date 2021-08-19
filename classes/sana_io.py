
# system packages
import os
import sys
import json

# installed packages
import numpy as np

# sana packages
from sana_geo import Polygon, Point

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
          "confidence": 1.0,
        }
    }
    return annotation
#
# end of anno_to_json

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
        return [], [], []

    # load the json data
    fp = open(ifile, 'r')
    data = json.loads(fp.read())

    # load the annotations
    # NOTE: this could be handled by a GeoJSON package?
    annotations = []
    class_names = []
    anno_names = []
    for annotation in data:

        # make sure the class name matches, if given
        if class_name is not None:

            # annotation has no class
            if 'classification' not in annotation['properties']:
                continue

            # annotation does not match given class name
            if annotation['properties']['classification']['name'] != class_name:
                continue
        #
        # end of class matching

        # load the list of coordinates list for this annotation
        # TODO: this is only needed cause of MultiPolygon,
        geo = annotation['geometry']
        poly = get_poly_from_geometry(geo)
        if poly is None:
            continue

        annotations.append(poly)

        # get the class name if it exists
        if 'classification' in annotation['properties']:
            class_names.append(
                annotation['properties']['classification']['name'])
        else:
            class_names.append('')

        # get the annotation name if it exists
        if 'name' in annotation['properties']:
            anno_names.append(
                annotation['properties']['name'])
        else:
            anno_names.append(
                'ROI_'+str(len(anno_names)))
    #
    # end of annotation loop

    return annotations, class_names, anno_names
#
# end of read_annotations

# generates a Polygon annotation from the geometry in a JSON annotation
def get_poly_from_geometry(geo):

    # TODO: this should be simplified.
    #        need to actually handle what a MultiPolygon is
    if geo['type'] == 'MultiPolygon':
        coords_list = geo['coordinates']
        for coords in coords_list:
            x = np.array([float(c[0]) for c in coords[0]])
            y = np.array([float(c[1]) for c in coords[0]])
            poly = Polygon(x, y, False, 0)
            if poly.area() > 100:
                return poly
    elif geo['type'] == 'Polygon':
        coords = geo['coordinates']
        x = np.array([float(c[0]) for c in coords[0]])
        y = np.array([float(c[1]) for c in coords[0]])
        return Polygon(x, y, False, 0)
    else:
        return None
#
# end of get_poly_from_geometry

# writes a list of Polygon annotations to a JSON annotation file
#  -ofile: location to write the annotations to
#  -annos: list of Polygon Annotations
#  -class_names: list of class names, blank if not given
#  -anno_names: list of anno names, blank if not given
def write_annotations(ofile, annos,
                      class_names=None, anno_names=None, confidences=None):

    # provide blank names if not given
    if class_names is None:
        class_names = ['']*len(annos)
    if anno_names is None:
        anno_names = ['']*len(annos)
    if confidences is None:
        confidences = [1.0]*len(annos)

    # loop through the Polygon annotations
    annotations = []
    for i in range(len(annos)):

        # convert to json format
        json_anno = anno_to_json(annos[i], class_names[i],
                                 anno_names[i], confidences[i])

        # store new annotation
        annotations.append(json_anno)
    #
    # end of Polygon annotations loop

    # write the file
    json.dump(annotations, open(ofile, 'w'))
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

# extracts the confidence value for each annotation stored in a JSON annotation
def read_confidences(ifile, class_name=None):

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
    confidences = []
    for annotation in data:

        # make sure the class name matches, if given
        if class_name is not None:

            # annotation has no class
            if 'classification' not in annotation['properties']:
                continue

            # annotation does not match given class name
            if annotation['properties']['classification']['name'] != class_name:
                continue
        #
        # end of class matching

        # get the confidence value, if doesn't exist output 100%
        if 'confidence' not in annotation['properties']:
            confidences.append(1.0)
        else:
            confidences.append(annotation['properties']['confidence'])
    #
    # end of annotation loop

    return confidences
#
# end of read_confidences

#
# end of file
