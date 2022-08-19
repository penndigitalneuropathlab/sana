#!/usr/bin/env python3

# system modules
import os
import sys
import argparse

# install modules
import numpy as np
from tqdm import tqdm

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

class DirectoryIncompleteError(Exception):
    def __init__(self, message='Directory is missing .csv file'):
        self.message = message
        super().__init__(self.message)

# stores INDD patient data an[<0;30;31Md Region data measured from the .svs slides
class Patient:
    def __init__(self, aid, is_hc):
        self.aid = aid
        self.is_hc = is_hc
        self.regions = {}
    #
    # end of constructor
    
    # TODO: this should read the header and then find the data, but works for now
    def get_subtype(self, pdata):
        for l in open(pdata, 'r'):
            parts = l.rstrip().split(',')
            if parts[2] == self.aid:
                return parts[14]
        return None
    def get_cohort(self, pdata):
        for l in open(pdata, 'r'):
            parts = l.rstrip().split(',')
            if parts[2] == self.aid:
                return parts[15]
        return None
    #
    # end of INDD data retrieval

    def add_data(self, entry):
        
        # create a new Region, if needed
        if entry.region not in self.regions:
            self.regions[entry.region] = Region(entry.region)

        # add the data to the Region
        self.regions[entry.region].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this patient
    def collapse(self):

        # collapse each region and store each antibody's measurement
        ao_data = {}
        for region in self.regions:
            region_ao_data = self.regions[region].collapse()
            for antibody in region_ao_data:
                if antibody not in ao_data:
                    ao_data[antibody] = []
                ao_data[antibody].append(region_ao_data[antibody])

        # calculate the average data over the regions in each antibody
        for antibody in ao_data:
            ao_data[antibody] = np.mean(ao_data[antibody], axis=0)

        return ao_data
    #
    # end of collapse
#
# end of Patient

# stores Subregions that were annotated and analyzed within the .svs slides
class Region:
    region_mapping = {
        'MFC': ['ROI',],
        'OFC': ['medOFC', 'latOFC',],
        'aCING': ['33', '24', '32',],
    }    
    def __init__(self, name):
        self.name = name
        self.subregions = {}
    #
    # end of constructor
    
    def get_subregion_name(self, roi_name):
        for subregion_name in self.region_mapping[self.name]:
            if subregion_name in roi_name:
                return subregion_name
        return None
    #
    # end of get_subregion_name

    def add_data(self, entry):
        
        # create a new Subregion, if needed
        subregion_name = self.get_subregion_name(entry.roi)
        if not subregion_name in self.subregions:
            self.subregions[subregion_name] = Subregion(subregion_name)
        
        # add the data to the subregion
        self.subregions[subregion_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this region
    def collapse(self):

        # collapse each subregion and store each antibody's measurement
        ao_data = {}
        for subregion in self.subregions:
            subregion_ao_data = self.subregions[subregion].collapse()
            for antibody in subregion_ao_data:
                if antibody not in ao_data:
                    ao_data[antibody] = []
                ao_data[antibody].append(subregion_ao_data[antibody])

        # calculate the average data over the subregions in each antibody
        for antibody in ao_data:
            ao_data[antibody] = np.mean(ao_data[antibody], axis=0)

        return ao_data
    #
    # end of collapse
#
# end of Region

# stores each Antibody that was available in this Subregion
class Subregion:
    def __init__(self, name):
        self.name = name
        self.antibodies = {}
    #
    # end of constructor
    
    def add_data(self, entry):

        # create a new Antibody, if needed
        if entry.antibody not in self.antibodies:
            self.antibodies[entry.antibody] = Antibody(entry.antibody)

        # add the data to the antibody
        self.antibodies[entry.antibody].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data in each antibody
    def collapse(self):
        ao_data = {}
        for antibody in self.antibodies:
            ao = self.antibodies[antibody].collapse()
            ao_data[antibody] = ao
        return ao_data
    #
    # end of collapse
#
# end of Subregion

# stores the measurements made in the ROI(s) from a Patient/Region/Subregion
# TODO: kinda weird that the antibody could technically store a entry with a different antibody
class Antibody:
    ao_measurements = {
        'NeuN': ['manual', 'auto', 'grn', 'pyr'],
        'SMI32': ['manual', 'auto'],
        'CALR6BC': ['manual', 'auto'],
        'parvalbumin': ['manual', 'auto'],    
        'SMI94': ['manual', 'auto', 'vert_fibers', 'horz_fibers'],
        'SMI35': ['manual', 'auto'],
    }
    
    def __init__(self, name):
        self.name = name
        self.rois = {}
    #
    # end of constructor

    def add_data(self, entry):
        self.rois[entry.roi] = entry

        self.get_ao_data()
    #
    # end of add_data

    # TODO: create functions for Signal, Densities, Sizes, etc...
    def get_ao_data(self):
        
        # collect the %AO measurements for each ROI
        self.ao_data = {}
        for roi_name in self.rois:
            data = []
            for ao_measurement in self.ao_measurements[self.name]:
                # TODO: if sub_aos is empty need to use _ao!!!!
                data.append(self.rois[roi_name].data[ao_measurement+'_sub_aos'])
            self.ao_data[roi_name] = np.array(data)
    #
    # end of get_ao_data

    # average all ROI data in this antibody into a single set of datapoints
    def collapse(self):
        ao = self.collapse_ao()

        return ao
    #
    # end of collapse

    def collapse_ao(self):

        # calculate the mean of the measurements over the ROIs        
        data = []
        for roi_name in self.rois:
            data.append(self.ao_data[roi_name])
        return np.mean(data, axis=0)
    #
    # end of collapse_ao
#
# end of Antibody

# loads and stores all data found within a ROI output directory
class Entry:
    def __init__(self, directory):
        self.directory = directory
        self.set_directory_parts()

        self.load_data()
    #
    # end of constructor
    
    def set_directory_parts(self):
        d, self.roi = os.path.split(self.directory)
        d, self.region = os.path.split(d)
        d, self.antibody = os.path.split(d)
        d, self.bid = os.path.split(d)
        
        self.params_f = [f for f in os.listdir(self.directory) if f.endswith('.csv')]
        if len(self.params_f) == 0:
            raise DirectoryIncompleteError
        self.params_f = self.params_f[0]
        self.slide_name = os.path.splitext(self.params_f)[0]
    #
    # end of get_directory_parts

    def load_data(self):
        
        # load the data stored in the params file
        self.params = Params()
        self.params.read_data(self.params_f)
        self.data = self.params.data

        # load extra data based on the antibody
        if self.antibody == 'NeuN':
            self.load_NeuN_data()
        elif self.antibody == 'SMI32':
            self.load_SMI32_data()
    #
    # end of load_data

    def load_NeuN_data(self):
        pass
    #
    # end of load_NeuN_data

    def load_SMI32_data(self):
        pass
    #
    # end of load_SMI32_data
#
# end of Entry

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
                rois = []
                for bid in bids:

                    # antibody or region doesn't exist for this ID 
                    if (not antibody in data[bid]) \
                       or (not region in data[bid][antibody]):
                        l = [np.nan]*11
                        a = [np.nan]*11
                        arr.append(l)
                        continue
                    
                    # loop through the ROI in each case
                    for roi in data[bid][antibody][region]:
                        
                        # retrieve the data
                        x = data[bid][antibody][region][roi]['%s_sub_aos' % measurement]
                        a = data[bid][antibody][region][roi]['sub_areas']
                        if x is None or a is None:
                            x, a = [], []
                        x, a = np.array(x), np.array(a)

                        # there was a problem generating this data
                        if any(np.isnan(x)):
                            l1, l2, l3, l4, l5, l6 = [np.nan]*6
                            l23, l56 = [np.nan]*2
                            l123, l456 = [np.nan]*2
                            l123456 = np.nan

                        # no layer annotations, just use the main AO
                        # TODO: this could also be blank, need to check above!
                        elif len(x) == 0:
                            l1, l2, l3, l4, l5, l6 = [np.nan]*6
                            l23, l56 = [np.nan]*2
                            l123, l456 = [np.nan]*2
                            l123456 = data[bid][antibody][region][roi]['%s_ao' % measurement]
                        
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
                        rois.append([bid, roi])
                    #
                    # end of rois loop
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

    return results, rois
#
# end of get_results

def write_results(odir, rois, results, mu, sigma):

    # write the headers
    long_ofile = os.path.join(odir, 'results_long.csv')
    long_fp = open(long_ofile, 'w')
    wide_ofile = os.path.join(odir, 'results_wide.csv')
    wide_fp = open(wide_ofile, 'w')
    long_fp.write(
        'AutopsyID,Antibody,Measurement,Region,ROI,Layer,AO,z_AO\n')
    wide_fp.write(
        'AutopsyID,Antibody,Measurement,Region,ROI,'+
        'L1,L2,L3,L4,L5,L6,L23,L56,L123,L456,L123456,'+
        'z_L1,z_L2,z_L3,z_L4,z_L5,z_L6,z_L23,z_L56,z_L123,z_L456,z_L123456\n')

    layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
              'L23', 'L56', 'L123', 'L456', 'L123456']

    # loop through the measurements
    for antibody in results:
        for measurement in results[antibody]:
            for region in results[antibody][measurement]:
                
                # loop through bids
                for i, (bid, roi) in enumerate(rois):

                    # get the x values and z scores for this roi
                    x = results[antibody][measurement][region][i]
                    z = (x - mu[antibody][measurement][region]) / \
                        sigma[antibody][measurement][region]

                    # write the long csv format
                    for j in range(len(x)):
                        long_fp.write('%s,%s,%s,%s,%s,%s,%.4g,%.4g\n' % \
                                      (bid, antibody, measurement, region, roi,
                                       layers[j], x[j], z[j]))

                    # write the wide csv format
                    wide_fp.write(('%s,%s,%s,%s,%s'+',%.4g'*22+'\n') % \
                                  (tuple((bid, antibody, measurement, region, roi)) \
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

def collect_patients(idir, hc):

    # load the list of HC patients
    hc = [l.rstrip() for l in open(hc, 'r')]

    patients = {}
    
    # loop through the BIDs found in the idir
    for bid in tqdm(os.listdir(idir)):
        bid_d = os.path.join(idir, bid)
        if not os.path.isdir(bid_d):
            continue

        # create a new Patient, if needed
        aid = '-'.join(bid.split('-')[:-1])        
        if aid not in patients:
            patients[aid] = Patient(aid, (aid in hc))
            
        # loop through the antibodies in the BID
        for antibody in os.listdir(bid_d):
            antibody_d = os.path.join(bid_d, antibody)
            if not os.path.isdir(antibody_d):
                continue
            
            # loop through the regions in this antibody
            for region in os.listdir(antibody_d):
                region_d = os.path.join(antibody_d, region)
                if not os.path.isdir(region_d):
                    continue
                
                # loop and store the ROIs
                for roi in os.listdir(region_d):
                    roi_d = os.path.join(region_d, roi)
                    if not os.path.isdir(roi_d):
                        continue

                    # load the data in this ROI
                    try:
                        entry = Entry(roi_d)
                    except DirectoryIncompleteError:
                        continue
                    
                    # add this ROI's data to the patient                    
                    patients[aid].add_data(entry)
                #
                # end of rois loop
            #
            # end of regions loop
        #
        # end of antibodies loop
    #
    # end of BIDs loop

    return patients
#
# end of collect_patients

def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # set of Patients that will be populated with the data in args.idir
    patients = collect_patients(args.idir, args.hc)

    # TODO: call function to generate curve plots
    #        various combinations of curves in the plots
    #         plots total, collapse subregion, collapse region
    
    # TODO: call function to generate box plots
    #        same combinations in curve plots
    #         also collapse combinations
    
    # TODO: call function to generate spreadsheets
    #        different AO data in each sheet in the spreadsheet?
    #        different collapse combinations in the spreadsheet?
    #         AID, REGION, SUBREGION, ROI, X_1, ..., X_6, X_23, X_56, X_123456
    #         AID, REGION, SUBREGION, N/A, X_1, ..., X_6, X_23, X_56, X_123456
    #         AID, REGION, N/A, N/A, X_1, ..., X_6, X_23, X_56, X_123456
    #         AID, N/A, N/A, N/A, X_1, ..., X_6, X_23, X_56, X_123456
    #               


    
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
    results, rois = get_results(data, bids, antibodies, regions)

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
                for i, (bid, roi) in enumerate(rois):
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
    write_results(args.odir, rois, results, mu, sigma)
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

