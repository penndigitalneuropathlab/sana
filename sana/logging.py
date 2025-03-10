
# system packages
import os
import sys
import logging
import dill

# installed packages
import numpy as np

# sana packages
from sana.geo import Point

# NOTE: the difference between 'full' and 'debug' is 'full' generates debugging plots
LEVEL_CONFIG = {
        'full': logging.DEBUG,
        'debug': logging.DEBUG,
        'normal': logging.INFO,
        'quiet': logging.ERROR,
}
class Logger():
    """ This class functions as a wrapper to the "logging" library, while also handling the I/O of various processing parameters that SANA determines
    :param debug_level: amount of debugging to output {full, debug, normal, quiet}
    :param fpath: path to where parameters and measurements should be written/read from
    :param name: name of the Logger instance
    """
    def __init__(self, debug_level, fpath="", name="SANA", read_data=True):
        self.debug_level = debug_level
        self.fpath = fpath
        self.name = name

        # configure the logger object
        self.logger = logging.getLogger(name)
        self.formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(filename)s :: <func> %(funcName)s :: Line: %(lineno)d :: %(message)s')

        # set the logging level
        self.debug_level = debug_level
        self.logger.setLevel(LEVEL_CONFIG[self.debug_level])

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

        # load the processing parameters
        if self.fpath != "" and read_data:
            self.data = self.read_data()
        else:
            self.data = {}

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
