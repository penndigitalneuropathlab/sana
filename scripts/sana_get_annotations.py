#!/usr/local/bin/python3.9

import os
import sys
import argparse

import sana_io

USAGE = os.path.expanduser(
    '~/neuropath/src/sana/scripts/usage/sana_get_annotations.usage')
DEF_RDIR = None

def main(argv):
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # get all the annotation files to process
    annotations = []
    for list_f in args.files:
        annotations += sana_io.read_list_file(list_f)

    if len(annotations) == 0 or args.target is None or args.odir is None:
        parser.print_usage()
        exit()

    # loop through the annotations
    for anno_if in annotations:

        # load the annos from the file matching the target class
        annos = sana_io.read_qupath_annotations(anno_if, name=args.target)

        # write the annos to the output
        anno_of = sana_io.get_ofname(
            anno_if, '.json', odir=args.odir, rdir=args.rdir)
        sana_io.write_qupath_annotations(anno_of, annos, name=args.target)
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser(usage=open(USAGE).read())
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument('-target', type=str,
                        help='target annotation class to get')
    parser.add_argument('-odir', type=str,
                        help='specify the location to write files to')
    parser.add_argument('-rdir', type=str, default=DEF_RDIR,
                        help='specify directory path to replace\n[default: ""]')
    return parser

#
# end of cmdl_parser

if __name__ == "__main__":
    main(sys.argv)
#
# end of file
