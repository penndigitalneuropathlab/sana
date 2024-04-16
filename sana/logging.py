
# system packages
import os
import sys
import logging
import dill

# installed packages
import numpy as np

# sana packages
from sana.geo import Point

# NOTE: to add a new field to an existing datatype:
#           1) add the key to X_KEYS list below

# define what our terminology for logging means in reference to the "logging" library
# NOTE: the difference between 'full' and 'debug' is 'full' generates debugging plots
LEVEL_CONFIG = {
        'full': logging.DEBUG,
        'debug': logging.DEBUG,
        'normal': logging.INFO,
        'quiet': logging.ERROR,
}
class Logger():
    """ This class functions as a wrapper to the "logging" library, while also handling the I/O of various processing parameters and measurements that SANA makes.
    :param debug_level: amount of debugging to output {full, debug, normal, quiet}
    :param fpath: path to where parameters and measurements should be written/read from
    :param name: name of the Logger instance
    """
    def __init__(self, debug_level, fpath="", name="SANA"):
        self.debug_level = debug_level
        self.fpath = fpath
        self.name = name

        # configure the logger object
        self.logger = logging.getLogger(name)
        self.formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(filename)s :: <func> %(funcName)s :: Line: %(lineno)d :: %(message)s')

        # set the logging level
        self.debug_level = debug_level
        self.logger.setLevel(LEVEL_CONFIG[self.debug_level])

        # decide if we're generating plots or not
        self.generate_plots = self.debug_level == 'full'

        # set the logging file handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        file_handler = logging.FileHandler('%s.log' % self.name, mode='w')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

        # set the logging to also output to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

        # load the parameters/measurements as a data dictionary
        if self.fpath != "":
            self.data = self.read_data()

    def read_data(self):
        if os.path.exists(self.fpath):
            with open(self.fpath, 'rb') as fp:
                return dill.load(fp)
        else:
            return {}
        
    def write_data(self):
        if self.fpath != "":
            with open(self.fpath, 'wb') as fp:
                dill.dump(self.data, fp)

    # wrapper function for logging messages
    def debug(self, s):
        self.logger.debug(s)
    def info(self, s):
        self.logger.info(s)
    def warning(self, s):
        self.logger.warning(s)
    def error(self, s):
        self.logger.error(s)