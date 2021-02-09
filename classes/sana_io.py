
import os
import sys

def get_fullpath(f):
    return os.path.abspath(os.path.expanduser(f))

def create_directory(f):
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))


#
# end of file
