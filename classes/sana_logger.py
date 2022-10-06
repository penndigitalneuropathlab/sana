#!/usr/bin/env python3

# system packages
import inspect
import logging
import sys

class SANALogger():
    def get_level_config(debug_level):
        # Dictionary of logging levels
        # full   --> generates plots
        # debug  --> outputs method checks and info (frame size, angle found, etc.)
        # normal --> just Processing Slide/Processing Frame; standard info
        # quiet  --> outputs nothing
        level_config = {'debug': logging.DEBUG,         # value: 10
                        'normal': logging.INFO,         # value: 20
                        # 'normal': logging.WARNING,     # value: 30
                        'quiet': logging.ERROR,        # value: 40
                        # 'critical': logging.CRITICAL # value: 50
                        }
        return level_config.get(debug_level)

    def get_sana_logger(debug_level):
        # gets file/line_num/method where logger is being called
        # ex. file: sana_io.py/line: 43/method: write_annotations()
        logger_name = inspect.stack()[1][3]

        # Configure logger object, creating a unique logger each time its called
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(filename)s :: <func> %(funcName)s :: Line: %(lineno)d :: %(message)s')
        
        # Set logging level from commandline
        level = SANALogger.get_level_config(debug_level.lower())
        logger.setLevel(level)

        # Setting logging file handler
        file_handler = logging.FileHandler('log.log',mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Setting logging output stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if level == logging.DEBUG:
            logger.plots = True
        else:
            logger.plots = False
        
        return logger

