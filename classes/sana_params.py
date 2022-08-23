
# system modules
import os

# installed modules
import numpy as np

# custom modules
from sana_geo import Point

# NOTE: to add a new field to an existing datatype:
#           1) add the key to X_KEYS list below

# NOTE: to add a new datatype:
#           1) create a X_KEYS list at top of file
#           2) add X_KEYS to the KEYS list below
#           3) create a parse_X() function
#           4) create a convert_X() function
#           5) add X_KEYS to conditional in parse_val()
#           6) add datatype check to to_string()

# lists of possible fields to store, separated by the datatype they should be stored as
INT_KEYS = [
    'lvl', 'padding',
    'csf_threshold',
    'manual_stain_threshold', 'auto_stain_threshold',
]
POINT_KEYS = [
    'loc', 'size',
    'crop_loc', 'crop_size',
    'ds',
]
M_KEYS = [
    'M1',
    'M2',
]
FLOAT_KEYS = [
    'angle1', 'angle2',    
    'area',
    'manual_ao', 'auto_ao',
    'vert_fibers_ao', 'horz_fibers_ao',
    'grn_ao', 'pyr_ao', 'tot_ao',
]
LIST_KEYS = [
    'sub_areas',
    'manual_sub_aos', 'auto_sub_aos',
    'vert_fibers_sub_aos', 'horz_fibers_sub_aos', 
    'grn_sub_aos', 'pyr_sub_aos', 'tot_sub_aos',
]
KEYS = INT_KEYS + POINT_KEYS + M_KEYS + FLOAT_KEYS + LIST_KEYS

# this class reads and writes the parameters and data associated with processed Frames
# TODO: Params isn't a great name since it also includes calculated data...
class Params:
    def __init__(self, fname=None):

        # initialize the data
        # TODO: may eventually want Params to inherit dict so that it doesn't have to store a dict
        # self.data = {}
        self.data = {k: None for k in KEYS}

        # line format in the output file
        self.line = '%s\t%s\n'

        if not fname is None:
            self.read_data(fname)
    #
    # end of constructor

    # reads the params/data from the .csv file
    def read_data(self, fname):
        self.fname = fname

        # make sure the file exists already
        if not os.path.exists(self.fname):
            return

        # load the fields into memory
        for line in open(self.fname, 'r'):
            key, val = line.split('\t', maxsplit=1)
            self.data[key] = self.parse_val(key, val.rstrip())
    #
    # end of read_data

    # loops through the stored key value pairs and writes them to the file
    def write_data(self, fname):
        fp = open(fname, 'w')
        for key in KEYS:
            if key in self.data:
                self.write_line(fp, key, self.data[key])
        fp.close()
    #
    # end of write_data

    # converts the value to a string then writes the data to a line
    def write_line(self, fp, key, val):
        val = self.to_string(val)
        fp.write(self.line % (key, val))
    #
    # end of write_line

    # parses the value from string to a specific datatype based on the key
    def parse_val(self, key, val):
        if len(val) == 0 or val is None or val == "None":
            return None
        elif key in INT_KEYS:
            return self.parse_int(val)
        elif key in FLOAT_KEYS:
            return self.parse_float(val)
        elif key in LIST_KEYS:
            return self.parse_list(val)
        elif key in POINT_KEYS:
            return self.parse_point(val)
        elif key in M_KEYS:
            return self.parse_M(val)
        else:
            return None
    #
    # end of parse_val

    # functions to parse strings into different datatypes
    def parse_int(self, x):
        return int(x)
    def parse_float(self, x):
        return float(x)
    def parse_list(self, x):
        return [float(y) for y in x.split('\t')]
    def parse_point(self, val):
        x0, x1 = [float(x) for x in val.split('\t')]
        return Point(x0, x1, False, 0)
    def parse_M(self, val):
        M = self.parse_list(val)
        return np.array(M, dtype=np.float64).reshape((2,3))
    #
    # end of parsing

    # converts a value into a string based on its datatype
    def to_string(self, x):
        if x is None:
            return None
        elif type(x) is str:
            return x
        elif type(x) is int or type(x) is np.int64 or type(x) is np.int32:
            return self.convert_int(x)
        elif type(x) is float or type(x) is np.float64 or type(x) is np.float32:
            return self.convert_float(x)
        elif type(x) is list or type(x) is Point:
            return self.convert_list(x)
        elif type(x) is np.ndarray and x.shape == (2,3):
            return self.convert_M(x)
        else:
            return None
    #
    # end of to_string

    # functions for converting values of different datatypes to strings
    def convert_int(self, x):
        return "%d" % (x)
    def convert_float(self, x):
        return '%.6f' % (x)
    def convert_list(self, x):
        return '\t'.join([self.to_string(y) for y in x])
    def convert_M(self, x):
        return self.convert_list(x.flatten())
    #
    # end of string conversion
#
# end of Params
