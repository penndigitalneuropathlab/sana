#!/usr/bin/env python3

import os
import argparse
import subprocess
import shlex
import numpy as np
import sana_io

parser = argparse.ArgumentParser()
parser.add_argument('-lists', type=str, nargs='*', required=True)
parser.add_argument('-njobs', type=int, required=True)
parser.add_argument('-job', type=str, required=True)
args = parser.parse_args()

slides = []
for list_f in args.lists:
    slides += sana_io.read_list_file(list_f)
if len(slides) == 0:
    print("**> No Slides Found")
    parser.print_usage()
    exit()

splits = np.array_split(slides, args.njobs)

d = os.path.join(os.path.dirname(list_f), 'tmp')
if not os.path.exists(d):
    os.makedirs(d)
files = [os.path.join(d, '%d_%s' % (i, os.path.basename(list_f))) for i in range(args.njobs)]

for i, x in enumerate(splits):
    fp = open(files[i], 'w')
    fp.write('\n'.join(x))
    fp.write('\n')
    fp.close()

job = open(args.job, 'r').read()
run_jobs = open(os.path.join(d, 'run_jobs.sh'), 'w')
for i, f in enumerate(files):
    #subprocess.Popen(job.replace('$1', f))
    
    of = os.path.join(d, '%d.sh') % i
    
    ojob = open(of, 'w')
    ojob.write('%s\n' % job.replace('$1', f.replace('\\', '/')))
    ojob.close()

    run_jobs.write('sh %s &\n' % of.replace('\\', '/'))
    
run_jobs.close()
