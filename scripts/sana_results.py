#!/usr/bin/env python3

# system modules
import os
import sys
import argparse
import random

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

LAYER_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
               'L23', 'L56', 'L123', 'L456', 'L123456']
LAYERS_HEADER = ''.join(['%s,' % x for x in LAYER_NAMES])
ZLAYERS_HEADER = ''.join(['z_%s,' % x for x in LAYER_NAMES])
FIELDS_HEADER = 'AutopsyID,BlockID,Region,Subregion,ROI,Antibody,Measurement,'
LONG_HEADER = FIELDS_HEADER+'Layer,AO,z_AO,'
WIDE_HEADER = FIELDS_HEADER+LAYERS_HEADER+ZLAYERS_HEADER
LONG_LINE = '%s,%s,%s,%s,%s,%s,%s,%s,%0.6f,%0.6f\n'
WIDE_LINE = '%s,%s,%s,%s,%s,%s,%s,'+'%0.6f,'*22+'\n'

ANTIBODY_MEASUREMENTS = {
    'NeuN': ['manual', 'auto', 'grn', 'pyr'],
    'SMI32': ['manual', 'auto'],
    'CALR6BC': ['manual', 'auto'],
    'parvalbumin': ['manual', 'auto'],    
    'SMI94': ['manual', 'auto', 'vert_fibers', 'horz_fibers'],
    'SMI35': ['manual', 'auto'],
}

class DirectoryIncompleteError(Exception):
    def __init__(self, message='Directory is missing .csv file'):
        self.message = message
        super().__init__(self.message)

class SubregionNameError(Exception):
    def __init__(self, region_name, roi_name, message='No Subregion could be selected with %s Region and %s ROI'):
        self.message = message % (region_name, roi_name)
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
        if entry.region_name not in self.regions:
            self.regions[entry.region_name] = Region(entry.region_name, entry.bid)

        # add the data to the Region
        self.regions[entry.region_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this patient
    def collapse_ao(self, antibody_name, measurement):

        data = []
        for region_name in self.regions:
            data.append(self.regions[region_name].collapse_ao(antibody_name, measurement))
        if len(data) != 0:
            return calc(np.mean, data)
        else:
            return np.full((11,), np.nan)
    #
    # end of collapse
#
# end of Patient

# stores Subregions that were annotated and analyzed within the .svs slides
class Region:
    region_mapping = {
        'MFC': ['ROI',],
        'OFC': [
            'medOFC', 'latOFC',
            's32',
            'med_GyrusRectus', 'lat_GyrusRectus',
            'med_MedialOrbitalGyrus', 'lat_MedialOrbitalGyrus'
        ],
        'aCING': ['a33', 'a32', 'a24a', 'a24b', 'a24c', 'a24'],
    }    
    def __init__(self, name, bid):
        self.name = name
        self.bid = bid
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
        if not entry.subregion_name in self.subregions:
            self.subregions[entry.subregion_name] = Subregion(entry.subregion_name)
        
        # add the data to the subregion
        self.subregions[entry.subregion_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this region
    def collapse_ao(self, antibody, measurement, subregions=[]):

        # collapse each subregion and store each antibody's measurement
        data = []
        for subregion_name in self.subregions:
            if len(subregions) != 0 and not subregion_name in subregions:
                continue
            x = self.subregions[subregion_name].collapse_ao(antibody, measurement)
            if not x is None:
                data.append(x)
        if len(data) != 0:
            return calc(np.mean, data)
        else:
            return np.full((11,), np.nan)
    #
    # end of collapse_ao
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
        if entry.antibody_name not in self.antibodies:
            self.antibodies[entry.antibody_name] = Antibody(entry.antibody_name)

        # add the data to the antibody
        self.antibodies[entry.antibody_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data in each antibody
    def collapse_ao(self, antibody_name, measurement):
        if antibody_name in self.antibodies:
            return self.antibodies[antibody_name].collapse_ao(measurement)
        else:
            return np.full((11,), np.nan)
    #
    # end of collapse_ao
#
# end of Subregion

# stores the measurements made in the ROI(s) from a Patient/Region/Subregion
# TODO: kinda weird that the antibody could technically store a entry with a different antibody
class Antibody:

    # TODO: it should be entires not ROIs, or Entry should be called ROI actually
    def __init__(self, name):
        self.name = name
        self.rois = {}
    #
    # end of constructor

    def add_data(self, entry):
        self.rois[entry.roi_name] = entry
    #
    # end of add_data

    # average all ROI data in this antibody into a single set of datapoints
    def collapse(self):
        ao = self.collapse_ao()

        return ao
    #
    # end of collapse

    def collapse_ao(self, measurement):

        # calculate the mean of the measurements over the ROIs        
        data = []
        for roi_name in self.rois:
            data.append(self.rois[roi_name].get_ao_data(measurement))
        if len(data) != 0:
            return calc(np.mean, data)
        else:
            return np.full((11,), np.nan)
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
        d, self.roi_name = os.path.split(self.directory)
        d, self.region_name = os.path.split(d)
        d, self.antibody_name = os.path.split(d)
        d, self.bid = os.path.split(d)

        region = Region(self.region_name, self.bid)
        self.subregion_name = region.get_subregion_name(self.roi_name)
        if self.subregion_name is None:
            raise SubregionNameError(self.region_name, self.roi_name)
        
        self.params_f = [f for f in os.listdir(self.directory) if f.endswith('.csv')]
        if len(self.params_f) == 0:
            raise DirectoryIncompleteError
        self.params_f = os.path.join(self.directory, self.params_f[0])
        self.slide_name = os.path.splitext(self.params_f)[0]
    #
    # end of get_directory_parts

    # TODO: create functions for Signal, Densities, Sizes, etc...
    def get_ao_data(self, measurement):

        # initialize array for storing all combinations of sublayer data
        # NOTE: 1, 2, 3, 4, 5, 6, 23, 56, 123, 456, 123456
        ao_data = np.full((11), np.nan)
        ao_data[-1] = self.data[measurement+'_ao']

        # get the subregion measurement data and the area of each sub ROI
        x = self.data[measurement+'_sub_aos']
        a = self.data['sub_areas']
        if x is None or a is None:
            x, a = [], []
        x, a = np.array(x), np.array(a)

        # no sub ROIs provided, just use the main measurement
        if len(x) == 0 or all(np.isnan(x)):
            pass
        # some NeuN data has all 6 layers annotated
        elif len(x) == 6:
            ao_data[0:6] = x
            ao_data[6] = (x[1:3] @ a[1:3]) / np.sum(a[1:3])
            ao_data[7] = (x[4:6] @ a[4:6]) / np.sum(a[4:6])
            ao_data[8] = (x[0:3] @ a[0:3]) / np.sum(a[0:3])
            ao_data[9] = (x[3:6] @ a[3:6]) / np.sum(a[3:6])
        # most data has 4 sub rois
        elif len(x) == 4:
            ao_data[0] = x[0]
            ao_data[6] = x[1]
            ao_data[3] = x[2]
            ao_data[7] = x[3]
            ao_data[8] = (x[0:2] @ a[0:2]) / np.sum(a[0:2])
            ao_data[9] = (x[2:4] @ a[2:4]) / np.sum(a[2:4])
        # CR data only had supra and infra
        elif len(x) == 2:
            ao_data[8:10] = x
        else:
            pass
        
        return ao_data
    #
    # end of get_ao_data
    
    def load_data(self):
        
        # load the data stored in the params file
        self.params = Params()
        self.params.read_data(self.params_f)
        self.data = self.params.data

        # load extra data based on the antibody
        if self.antibody_name == 'NeuN':
            self.load_NeuN_data()
        elif self.antibody_name == 'SMI32':
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
                    except (DirectoryIncompleteError, SubregionNameError):
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

def get_HC_model(patients, antibody_name, measurement, region_name=None, subregion_name=None):
    hc = []
    for aid in patients:

        # get only HC patients
        patient = patients[aid]
        if not patient.is_hc:
            continue

        # no region specified, calculate model over entire patient
        if region_name is None:
            hc.append(patient.collapse_ao(antibody_name, measurement))
        else:

            # region not available for this patient
            if not region_name in patient.regions:
                continue

            # no subregion specified, calculate model over region
            region = patient.regions[region_name]
            if subregion_name is None:
                hc.append(region.collapse_ao(antibody_name, measurement))
            else:

                # subregion not available for this patient
                if not subregion_name in region.subregions:
                    continue

                # calculate model over the subregion
                subregion = region.subregions[subregion_name]
                hc.append(subregion.collapse_ao(antibody_name, measurement))
    #
    # end of patients loop
    
    # generate the model over the HC patients
    if len(hc) != 0:
        return calc(np.mean, hc), calc(np.std, hc)
    else:
        return np.full((11,), np.nan), np.full((11,), np.nan)
#
# end of get_HC_model

def calc(func, x, debug=False):
    if type(x) is list:
        x = np.array(x)
    y = np.full((x.shape[1],), np.nan)
    for i in range(x.shape[1]):
        if debug:
            print('\n', i)            
            print(x[:,i])
            print(~np.isnan(x[:,i]))
            print(x[~np.isnan(x[:,i]), i])
        if all(np.isnan(x[:,i])):
            continue
        y[i] = func(x[~np.isnan(x[:,i]), i], axis=0)
        if debug:
            print(y[i])
    return y

def z_score(x, model):
    mu, sigma = model
    return (x-mu) / sigma
#
# end of z_score

def generate_spreadsheets(odir, patients):

    long_fp = open(os.path.join(odir, 'long_results.csv'), 'w')
    long_fp.write('%s\n' % LONG_HEADER)
    wide_fp = open(os.path.join(odir, 'wide_results.csv'), 'w')
    wide_fp.write('%s\n' % WIDE_HEADER)

    region_models = {}
    subregion_models = {}
    for antibody_name in ANTIBODY_MEASUREMENTS:
        for measurement in ANTIBODY_MEASUREMENTS[antibody_name]:
            patient_model = get_HC_model(
                patients, antibody_name, measurement)
            for aid in sorted(patients.keys()):
                patient = patients[aid]
                for region_name in sorted(patient.regions.keys()):
                    region = patient.regions[region_name]
                    bid = region.bid
                    
                    # calcualte the HC model for this region
                    if region_name not in region_models:
                        region_models[region_name] = get_HC_model(
                            patients, antibody_name, measurement,
                            region_name)
                    for subregion_name in sorted(region.subregions.keys()):
                        subregion = region.subregions[subregion_name]
                        if not antibody_name in subregion.antibodies:
                            continue
                        antibody = subregion.antibodies[antibody_name]
                        
                        # calculate the HC model for this subregion
                        if region_name not in subregion_models:
                            subregion_models[region_name] = {}
                        if subregion_name not in subregion_models[region_name]:
                            subregion_models[region_name][subregion_name] = \
                                get_HC_model(patients, antibody_name, measurement,
                                             region_name, subregion_name)
                        
                        # write each individual ROI data row
                        for roi_name in sorted(antibody.rois.keys()):
                            x = antibody.rois[roi_name].get_ao_data(measurement)
                            z = z_score(
                                x, subregion_models[region_name][subregion_name])
                            write_row(long_fp, wide_fp, aid, bid,
                                      region_name, subregion_name, roi_name,
                                      antibody_name, measurement, x, z)
                        #
                        # end of ROIs loop

                        # write a row for the average over the ROIs in the subregion
                        x = subregion.collapse_ao(antibody_name, measurement)
                        z = z_score(
                            x, subregion_models[region_name][subregion_name])
                        write_row(long_fp, wide_fp, aid, bid,
                                  region_name, subregion_name, 'AVG',
                                  antibody_name, measurement, x, z)
                    #
                    # end of subregions loop

                    # write a row for the average over the subregions in the region
                    x = region.collapse_ao(antibody_name, measurement)
                    z = z_score(x, region_models[region_name])
                    write_row(long_fp, wide_fp, aid, bid,
                              region_name, 'AVG', 'AVG',
                              antibody_name, measurement, x, z)

                    # special case: combines a24(a-c) subregions as a24 for aCING
                    if region.name == 'aCING':
                        x = region.collapse_ao(antibody_name, measurement,
                                               subregions=['a24a', 'a24b', 'a24c'])
                        z = z_score(x, region_models[region_name])
                        write_row(long_fp, wide_fp, aid, bid,
                                  region_name, 'a24abc', 'AVG',
                                  antibody_name, measurement, x, z)
                #
                # end of regions loop

                # write a row for the average over all regions
                x = patient.collapse_ao(antibody_name, measurement)
                z = z_score(x, patient_model)
                write_row(long_fp, wide_fp, aid, bid,
                          'AVG', 'AVG', 'AVG',
                          antibody_name, measurement, x, z)
            #
            # end of aids loop
        #
        # end of measurements loop
    #
    # end of antibodies loop

    # print('PATIENT:\n',patient_model)
    # for region in region_models:
    #     print('%s\n' % region, region_models[region])
    # for region in subregion_models:
    #     for subregion in subregion_models[region]:
    #         print('%s\n' % subregion, subregion_models[region][subregion])
#
# end of generate_spreadsheets

def write_row(long_fp, wide_fp, aid, bid, region_name, subregion_name, roi_name,
              antibody_name, measurement, x, z):
    for i in range(len(x)):
        long_fp.write(
            LONG_LINE % \
            (aid, bid, region_name, subregion_name, roi_name,
             antibody_name, measurement, LAYER_NAMES[i], x[i], z[i]))
    wide_fp.write(
        WIDE_LINE % \
        (aid, bid, region_name, subregion_name, roi_name, antibody_name, measurement,
         *tuple(x), *tuple(z)))
#
# end of write_row
    
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # set of Patients that will be populated with the data in args.idir
    patients = collect_patients(args.idir, args.hc)

    generate_spreadsheets(args.odir, patients)
    
    # TODO: call function to generate curve plots
    #        various combinations of curves in the plots
    #         plots total, collapse subregion, collapse region
    
    # TODO: call function to generate box plots
    #        same combinations in curve plots
    #         also collapse combinations
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

