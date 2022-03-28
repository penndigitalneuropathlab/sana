#!/usr/bin/env python3

# system modules
import os
import sys
import argparse

# install modules
import numpy as np

# custom modules
import sana_io
from sana_loader import Loader
from sana_frame import Frame
from sana_params import Params

# debugging modules
from matplotlib import pyplot as plt
from matplotlib.pyplot import setp
from matplotlib.lines import Line2D
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("Set2"))

MEASUREMENTS = {
    'parvalbumin': ['manual', 'auto'],
    'SMI94': ['manual', 'auto', 'vert_fibers', 'horz_fibers'],    
}

def get_cohort(id, stats):
    for x in stats:
        parts = x.split(',')
        aid, grp, sub, dis = parts[3], parts[13], parts[14], parts[15]
        if '-'.join(aid.split('-')[:2]) == '-'.join(id.split('-')[:2]):
            return grp

def load_data(idir, aid, antibody, region, roi):

    # build the filepath
    d = os.path.join(idir, aid, antibody, region, roi)
    f = [x for x in os.listdir(d) if x.endswith('.csv')][0]
    f = os.path.join(d, f)

    # load the data
    params = Params()
    params.read_data(f)
    data = params.data
    
    return data
#
# end of load_data

def get_results(data, bids, antibodies, regions):

    # loop through the antibodies
    results = {}
    for antibody in antibodies:
        results[antibody] = {}

        # loop through the measurements for each antibody
        for measurement in MEASUREMENTS[antibody]:
            results[antibody][measurement] = {}
            
            # loop through the regions for each antibody
            for region in regions:
                
                # loop through all IDs
                arr = []
                for bid in bids:

                    # antibody or region doesn't exist for this ID 
                    if (not antibody in data[bid]) \
                       or (not region in data[bid][antibody]):
                        l = [np.nan]*11
                        a = [np.nan]*11
                        arr.append(l)
                        continue

                    # retrieve the data
                    # TODO: what if sub_aos doesn't exist? just use main_ao
                    x = data[bid][antibody][region]['%s_sub_aos' % measurement]
                    a = data[bid][antibody][region]['sub_areas']
                    x, a = np.array(x), np.array(a)

                    # there was a problem generating this data
                    if any(np.isnan(x)) or len(x) == 0:
                        l1, l2, l3, l4, l5, l6 = [np.nan]*6
                        l23, l56 = [np.nan]*2
                        l123, l456 = [np.nan]*2
                        l123456 = np.nan

                    # full layer annotations
                    elif len(x) == 6:
                        l1, l2, l3, l4, l5, l6 = x
                        l23 = (x[1:3] @ a[1:3]) / np.sum(a[1:3])
                        l56 = (x[4:6] @ a[4:6]) / np.sum(a[4:6])
                        l123 = (x[0:3] @ a[0:3]) / np.sum(a[0:3])
                        l456 = (x[3:6] @ a[3:6]) / np.sum(a[3:6])
                        l123456 = (x[0:6] @ a[0:6]) / np.sum(a[0:6])

                    # partial annotations
                    elif len(x) == 4:
                        l1, l2, l3 = [x[0], np.nan, np.nan]
                        l4, l5, l6 = [x[2], np.nan, np.nan]
                        l23, l56 = x[1], x[3]
                        l123 = (x[0:2] @ a[0:2]) / np.sum(a[0:2])
                        l456 = (x[2:4] @ a[2:4]) / np.sum(a[2:4])
                        l123456 = (x[0:4] @ a[0:4]) / np.sum(a[0:4])

                    # only used in CR right now
                    elif len(x) == 2:
                        l1, l2, l3, l4, l5, l6 = [np.nan]*6
                        l23, l56 = [np.nan]*2
                        l123 = x[0]
                        l456 = x[1]
                        l123456 = (x[0:2] @ a[0:2]) / np.sum(a[0:2])

                    # store the subregions and combination of subregions
                    x = [l1, l2, l3, l4, l5, l6, l23, l56, l123, l456, l123456]
                    arr.append(x)
                
                #
                # end of bids loop
            
                # finally, store the results forthis measurement/antibody/region
                results[antibody][measurement][region] = np.array(arr)
            #
            # end of regions loop
        #
        # end of measurement loop
    #
    # end of antibody loop

    return results
#
# end of get_results

def write_results(odir, bids, results, mu, sigma):

    # write the headers
    long_ofile = os.path.join(odir, 'results_long.csv')
    long_fp = open(long_ofile, 'w')
    wide_ofile = os.path.join(odir, 'results_wide.csv')
    wide_fp = open(wide_ofile, 'w')    
    long_fp.write(
        'AutopsyID,Antibody,Measurement,Region,Layer,AO,z_AO\n')
    wide_fp.write(
        'AutopsyID,Antibody,Measurement,Region,'+
        'L1,L2,L3,L4,L5,L6,L23,L56,L123,L456,L123456,'+
        'z_L1,z_L2,z_L3,z_L4,z_L5,z_L6,z_L23,z_L56,z_L123,z_L456,z_L123456\n')

    layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
              'L23', 'L56', 'L123', 'L456', 'L123456']

    # loop through the measurements
    for antibody in results:
        for measurement in results[antibody]:
            for region in results[antibody][measurement]:
                
                # loop through bids
                for i, bid in enumerate(bids):

                    # get the x values and z scores for this roi
                    x = results[antibody][measurement][region][i]
                    z = (x - mu[antibody][measurement][region]) / \
                        sigma[antibody][measurement][region]

                    # write the long csv format
                    for j in range(len(x)):
                        long_fp.write('%s,%s,%s,%s,%s,%s,%.4g,%.4g\n' % \
                                      (aid, antibody, measurement, region,
                                       layers[j], x[j], z[j]))

                    # write the wide csv format
                    wide_fp.write(('%s,%s,%s,%s,%s'+',%.4g'*22+'\n') % \
                                  (tuple((aid, antibody, measurement, region)) \
                                   + tuple(x) + tuple(z)))
                #
                # end of bids loop
            #
            # end of regions loop
        #
        # end of measurements loop
    #
    # end of antibodies loop
#
# end of write_results
                
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    data = {}

    # loop through the IDs
    for bid in os.listdir(args.idir):
        d = os.path.join(args.idir, bid)
        if not os.path.isdir(d):
            continue
        data[bid] = {}
        
        # loop through the antibodies in the ID
        for antibody in os.listdir(d):
            d = os.path.join(args.idir, bid, antibody)
            if not os.path.isdir(d):
                continue
            data[bid][antibody] = {}
            
            # loop through the regions
            for region in os.listdir(d):
                d = os.path.join(args.idir, bid, antibody, region)
                if not os.path.isdir(d):
                    continue
                data[bid][antibody][region] = []
                
                # loop and store the rois
                for roi in os.listdir(d):
                    d = os.path.join(args.idir, bid, antibody, region, roi)
                    if not os.path.isdir(d):
                        continue

                    # finally, load the .csv file and store it
                    x = load_data(args.idir, bid, antibody, region, roi)
                    data[bid][antibody][region] = x
                    
                    # TODO: handle multiple ROIs
                    # data[bid][antibody][region].append(x)
                    


    # get all the block IDs in the results
    bids = sorted(list(iter(data.keys())))
            
    # get all the regions and antibodies in the results
    regions, antibodies = [], []
    for bid in data:
        for antibody in data[bid]:
            if not antibody in antibodies:
                antibodies.append(antibody)        
            for region in data[bid][antibody]:
                if not region in regions:
                    regions.append(region)
    regions = sorted(regions)
    antibodies = sorted(antibodies)
    
    # reformat the data into an array
    results = get_results(data, bids, antibodies, regions)

    # get the HC cases
    hc_aids = [l.rstrip() for l in open(args.hc, 'r')]

    # loop through antibodies
    mu, sigma = {}, {}
    for antibody in results:

        # loop through measurements for each antibody
        mu[antibody] = {}
        sigma[antibody] = {}
        for measurement in results[antibody]:

            # loop through the region for each measurement
            mu[antibody][measurement] = {}
            sigma[antibody][measurement] = {}            
            for region in results[antibody][measurement]:
                
                # get all s associated with HC IDs
                hc = []
                for i, bid in enumerate(bids):
                    aid = '-'.join(bid.split('-')[:2])
                    if aid in hc_aids:
                        hc.append(results[antibody][measurement][region][i])
                hc = np.array(hc)

                # calculate the mean and stdev over each column
                m, s = np.zeros(hc.shape[1]), np.zeros(hc.shape[1])
                for i in range(hc.shape[1]):

                    # entire column is nan, m and s are nan
                    if all(np.isnan(hc[:,i])):
                        m[i] = np.nan
                        s[i] = np.nan

                        # some of the column is nan, remove from the calculation
                    else:
                        x = hc[:,i]
                        x = x[~np.isnan(x)]
                        m[i] = np.mean(x)
                        s[i] = np.std(x)
                    mu[antibody][measurement][region] = m
                    sigma[antibody][measurement][region] = s
                #
                # end of HC profile calculation
            #
            # end of regions loop
        #
        # end of measurements loop
    #
    # end of antibodies loop

    # finally, write the results        
    write_results(args.odir, bids, results, mu, sigma)
#
# end of main

def cmdl_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-idir', type=str, required=True,
        help="directory path containing output .csv files")
    parser.add_argument(
        '-odir', type=str, required=True,
        help="directory path containing to write the output files")
    parser.add_argument(
        '-hc', type=str, required=True,
        help="file containing a list of HC cases")
    return parser
#
# end of cmdl_parser

if __name__ == '__main__':
    main(sys.argv)
    
#
# end of file

