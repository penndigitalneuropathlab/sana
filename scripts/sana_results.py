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
               'L23', 'L56', 'L123', 'L456', 'L23456', 'L123456']
LAYERS_HEADER = ''.join(['%s,' % x for x in LAYER_NAMES])
ZLAYERS_HEADER = ''.join(['z_%s,' % x for x in LAYER_NAMES])
FIELDS_HEADER = 'SlideName,AutopsyID,BlockID,Hemisphere,Region,Subregion,ROI,Antibody,Measurement,'
LONG_HEADER = FIELDS_HEADER+'Layer,AO,z_AO,'
WIDE_HEADER = FIELDS_HEADER+LAYERS_HEADER+ZLAYERS_HEADER
LONG_LINE = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%0.16f,%0.16f\n'
WIDE_LINE = '%s,%s,%s,%s,%s,%s,%s,%s,%s,'+'%0.16f,'*(2*len(LAYER_NAMES))+'\n'

# TODO: rename grn to non
ANTIBODY_MEASUREMENTS = {
    'NeuN': ['manual', 'auto', 'grn', 'pyr'],
    'SMI32': ['manual', 'auto'],
    'CALR6BC': ['manual', 'auto'],
    'parvalbumin': ['manual', 'auto'],    
    'SMI94': ['manual', 'auto', 'vert_fibers', 'horz_fibers'],
    'SMI35': ['manual', 'auto'],
    'MJFR13': ['manual', 'auto', 'lb_wc'],
    'AT8': ['manual', 'auto'],
    'TDP43': ['manual', 'auto'],
    'TDP43MP': ['manual', 'auto'],
    'GFAP': ['manual', 'auto'],
    'IBA1': ['manual', 'auto'],
    'MeguroTriton': ['manual', 'auto'],
    'Ferritin': ['manual', 'auto'],
}

class DirectoryIncompleteError(Exception):
    def __init__(self, message='Directory is missing .csv file'):
        self.message = message
        super().__init__(self.message)

class SubregionNameError(Exception):
    def __init__(self, region_name, roi_name, message='No Subregion could be selected with %s Region and %s ROI'):
        self.message = message % (region_name, roi_name)
        super().__init__(self.message)

# stores INDD patient data and Region data measured from the .svs slides
class Patient:
    def __init__(self, aid, is_hc):
        self.aid = aid
        self.is_hc = is_hc
        self.hemispheres = {}
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
        
        # create a new Hemisphere, if needed
        if entry.hemisphere_name not in self.hemispheres:
            self.hemispheres[entry.hemisphere_name] = Hemisphere(entry.hemisphere_name)

        # add the data to the Hemisphere
        self.hemispheres[entry.hemisphere_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this patient
    def collapse_ao(self, antibody_name, measurement):

        data = []
        for hemisphere_name in self.hemispheres:
            data.append(self.hemispheres[hemisphere_name].collapse_ao(antibody_name, measurement))
        if len(data) != 0:
            return calc(np.mean, data)
        else:
            return np.full((len(LAYER_NAMES),), np.nan)
    #
    # end of collapse_ao
#
# end of Patient

class Hemisphere:
    def __init__(self, name):
        self.name = name
        self.regions = {}
        
    def add_data(self, entry):

        # create a new Region, if needed
        if entry.region_name not in self.regions:
            self.regions[entry.region_name] = Region(entry.region_name, entry.bid)

        # add the data to the Region
        self.regions[entry.region_name].add_data(entry)
    #
    # end of add_data

    # gets the average ROI data over this hemisphere
    def collapse_ao(self, antibody_name, measurement):
        
        data = []
        for region_name in self.regions:
            data.append(self.regions[region_name].collapse_ao(antibody_name, measurement))
        if len(data) != 0:
            return calc(np.mean, data)
        else:
            return np.full((len(LAYER_NAMES),), np.nan)
    #
    # end of collapse_ao
#
# end of Hemisphere

# stores Subregions that were annotated and analyzed within the .svs slides
class Region:
    region_mapping = {
        #'MFC': ['MFC_crown', 'MFC_sulcus'],
        'MFC': ['MFC_'],
        'OFC': [
            'medOFC', 'latOFC',
            's32',
            'med_GyrusRectus', 'lat_GyrusRectus',
            'med_MedialOrbitalGyrus', 'lat_MedialOrbitalGyrus',
            'lat_Olfactory_sulcus', 'med_Olfactory_sulcus',
            'Rostral_sulcus', 'Orbital_sulcus',
            'Cingulate_sulcus'
        ],
        'aCING': ['a33', 'a32', 'a24a', 'a24b', 'a24c', 'a24'],
        'SMTC': [],
        'M1': [],
        'ANG': [],
    }
    for region_name in region_mapping:
        region_mapping[region_name] += ['ROI', 'Greatest GM Sampling', 'GM_SEG']
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
            return np.full((len(LAYER_NAMES),), np.nan)
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
            return np.full((len(LAYER_NAMES),), np.nan)
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
            return np.full((len(LAYER_NAMES),), np.nan)
    #
    # end of collapse_ao
#
# end of Antibody

# loads and stores all data found within a ROI output directory
class Entry:
    def __init__(self, patients, directory, hemisphere_data, hc_aids):
        self.directory = directory
        
        self.set_slide_parts(hemisphere_data)
        
        self.load_data()

        if not self.aid in patients:
            patients[self.aid] = Patient(self.aid, self.aid in hc_aids)
        patients[self.aid].add_data(self)
    #
    # end of constructor
    
    def set_slide_parts(self, hemisphere_data):
        d, self.roi_name = os.path.split(self.directory)
        d, self.slide_name = os.path.split(d)
        self.slide_f = self.slide_name+'.svs'
        try:
            self.region_name = sana_io.get_region(self.slide_f)
            self.antibody_name = sana_io.get_antibody(self.slide_f)
        except:
            print(self.slide_f)
        self.bid = sana_io.get_bid(self.slide_f)        
        self.aid = sana_io.get_aid(self.slide_f)

        self.roi_name = self.slide_name+'_'+self.roi_name
        
        if self.region_name == 'CING':
            self.region_name = 'aCING'

        region = Region(self.region_name, self.bid)
        self.subregion_name = region.get_subregion_name(self.roi_name)
        if self.subregion_name is None:
            raise SubregionNameError(self.region_name, self.roi_name)

        self.params_f = os.path.join(self.directory, self.slide_name+'.csv')
        if not os.path.exists(self.params_f):
            raise DirectoryIncompleteError

        if self.bid in hemisphere_data:
            self.hemisphere_name = hemisphere_data[self.bid]
        else:
            self.hemisphere_name = sana_io.get_hemi(self.slide_f)
    #
    # end of get_slide_parts

    # TODO: create functions for Signal, Densities, Sizes, etc...
    def get_ao_data(self, measurement):

        # initialize array for storing all combinations of sublayer data
        # NOTE: 1, 2, 3, 4, 5, 6, 23, 56, 123, 456, 23456, 123456
        ao_data = np.full((12), np.nan)
        ao_data[-1] = self.data[measurement+'_ao']

        # sometimes the sub_aos are not given
        if measurement+'_sub_aos' not in self.data:
            return ao_data
                
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
            if np.isnan(x[3]):
                inds = np.array([0,1,2,4,5])
            else:
                inds = np.array([0,1,2,3,4,5])
            ao_data[10] = (x[inds][1:] @ a[inds][1:]) / np.sum(a[inds][1:])
            ao_data[-1] = (x[inds] @ a[inds]) / np.sum(a[inds])
        # most data has 4 sub rois
        elif len(x) == 4:
            ao_data[0] = x[0]
            ao_data[6] = x[1]
            ao_data[3] = x[2]
            ao_data[7] = x[3]
            ao_data[8] = (x[0:2] @ a[0:2]) / np.sum(a[0:2])
            ao_data[9] = (x[2:4] @ a[2:4]) / np.sum(a[2:4])
            if np.isnan(x[2]):
                inds = np.array([0,1,3])
            else:
                inds = np.array([0,1,2,3])
            ao_data[10] = (x[inds][1:] @ a[inds][1:]) / np.sum(a[inds][1:])
            ao_data[-1] = (x[inds] @ a[inds]) / np.sum(a[inds])
        # CR data only had supra and infra
        elif len(x) == 2:
            ao_data[8:10] = x
            ao_data[-1] = (x @ a) / np.sum(a)
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

# TODO: why not make the output directory structure: aid/hemi/region/roi/
def collect_patients(idir, hc, hemi, exclude):

    # load the list of HC patients
    if os.path.exists(hc):
        hc = [l.rstrip() for l in open(hc, 'r')]
    else:
        hc = []

    hemisphere_data = {}        
    if os.path.exists(hemi):
        hemi_data = [l.rstrip().split(',') for l in open(hemi, 'r')]
        for x in hemi_data:
            hemisphere_data[x[1]] = x[2]

    exclude_slides = []
    if os.path.exists(exclude):
        exclude_slides = [l.rstrip() for l in open(exclude, 'r')]
    print(exclude_slides)
    patients = {}
    antibodies = []
    
    # loop through the BIDs found in the idir
    for slide_name in tqdm(os.listdir(idir)):
        slide_d = os.path.join(idir, slide_name)
        if not os.path.isdir(slide_d):
            continue

        # skip the files that are deemed to be unworthy of analyzing
        if slide_name in exclude_slides:
            continue
        
        # loop through the ROIs in this region
        for roi in os.listdir(slide_d):
            roi_d = os.path.join(slide_d, roi)
            if not os.path.isdir(roi_d):
                continue

            # load the data in this ROI
            try:
                entry = Entry(patients, roi_d, hemisphere_data, hc)
            except (DirectoryIncompleteError, SubregionNameError) as e:
                print(e)
                continue
                    
            if entry.antibody_name not in antibodies:
                antibodies.append(entry.antibody_name)
        #
        # end of rois loop
    #
    # end of slides loop

    return patients, antibodies
#
# end of collect_patients

def get_HC_model(patients, antibody_name, measurement, hemisphere_name=None, region_name=None, subregion_name=None):

    # loop through the HC patients
    hc = []
    hc_aids = [aid for aid in patients if patients[aid].is_hc]
    for aid in hc_aids:
        patient = patients[aid]
        
        # no hemisphere specified, calculate model over entire patient
        if hemisphere_name is None:
            hc.append(patient.collapse_ao(antibody_name, measurement))
            continue

        # hemisphere not available for this patient, skip!
        if not hemisphere_name in patient.hemispheres:
            continue
            
        # no region specified, calculate model over hemisphere
        hemisphere = patient.hemispheres[hemisphere_name]
        if region_name is None:
            hc.append(hemisphere.collapse_ao(antibody_name, measurement))
            continue

        # region not available for this patient, skip!
        if not region_name in hemisphere.regions:
            continue

        # no subregion specified, calculate model over region
        region = hemisphere.regions[region_name]
        if subregion_name is None:
            hc.append(region.collapse_ao(antibody_name, measurement))
            continue

        # subregion not available for this patient, skip!
        if not subregion_name in region.subregions:
            continue

        # calculate model over the subregion
        subregion = region.subregions[subregion_name]
        hc.append(subregion.collapse_ao(antibody_name, measurement))
    #
    # end of patients loop

    # generate the model over the HC patients
    if len(hc) != 0:
        mu, sg = calc(np.mean, hc), calc(np.std, hc)
    else:
        mu, sg = np.full((len(LAYER_NAMES),), np.nan), np.full((len(LAYER_NAMES),), np.nan)

    return mu, sg
#
# end of get_HC_model

# this function applies a input function over a (N,M) input array over the N axis
#  it makes sure to only aggregate data in each column that is not nan, this ensure
#  that our HC model will never be nan for example (unless entire cohort is empty)
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

def generate_files(odir, prefix):
    long_fp = open(os.path.join(odir, '%s_long_results.csv' % prefix), 'w')
    long_fp.write('%s\n' % LONG_HEADER)
    wide_fp = open(os.path.join(odir, '%s_wide_results.csv' % prefix), 'w')
    wide_fp.write('%s\n' % WIDE_HEADER)
    
    return long_fp, wide_fp

# this function collapses our hierarchical patient dataset into a table
def generate_spreadsheets(odir, patients, available_antibodies):

    # generate the file pointers for the raw and aggregated data
    roi_fps = generate_files(odir, 'full')
    subregion_fps = generate_files(odir, 'agg_roi')
    region_fps = generate_files(odir, 'agg_subregion')
    hemisphere_fps = generate_files(odir, 'agg_region')
    patient_fps = generate_files(odir, 'agg_hemisphere')
    models_fp = open(os.path.join(odir, 'HC_models.csv'), 'w')
    models_fp.write('%s\n' % WIDE_HEADER)
    
    # this dictionary stores the individual HC models for each combination
    #  of antibody, measurement, hemisphere, region, and subregion
    hc_models = {}

    # loop through antibodies to aggregate
    for antibody_name in available_antibodies:
        hc_models[antibody_name] = {}
        
        # loop through measurements in each antibody
        for measurement in ANTIBODY_MEASUREMENTS[antibody_name]:
            
            # get HC model for entire cohort
            patient_model = get_HC_model(patients, antibody_name, measurement)
            hc_models[antibody_name][measurement] = {
                'model': patient_model,
            }
            hemisphere_models = hc_models[antibody_name][measurement]
            
            # loop through patients
            for aid in sorted(patients.keys()):
                bid = None
                
                # loop through hemispheres in this patient
                patient = patients[aid]
                for hemisphere_name in sorted(patient.hemispheres.keys()):
                    hemisphere = patient.hemispheres[hemisphere_name]

                    # calculate the HC model for this hemisphere, if not already generated
                    if hemisphere_name not in hemisphere_models:
                        hemisphere_models[hemisphere_name] = {
                            'model': get_HC_model(
                                patients, antibody_name, measurement,
                                hemisphere_name,
                            )
                        }
                    region_models = hemisphere_models[hemisphere_name]
                
                    # loop through regions in this hemisphere
                    for region_name in sorted(hemisphere.regions.keys()):
                        region = hemisphere.regions[region_name]
                        bid = region.bid
                    
                        # calculate the HC model for this region, if not already generated
                        if region_name not in region_models:
                            region_models[region_name] = {
                                'model': get_HC_model(
                                    patients, antibody_name, measurement,
                                    hemisphere_name, region_name,
                                )
                            }
                        subregion_models = region_models[region_name]

                        # loop through subregions in this region
                        for subregion_name in sorted(region.subregions.keys()):
                            subregion = region.subregions[subregion_name]

                            # antibody not found in this subregion, we skip
                            if not antibody_name in subregion.antibodies:
                                continue

                            # calculate the HC model for this subregionm, if needed
                            if subregion_name not in subregion_models:
                                subregion_models[subregion_name] = {
                                    'model': get_HC_model(
                                        patients, antibody_name, measurement,
                                        hemisphere_name, region_name, subregion_name,
                                    )
                                }

                            antibody = subregion.antibodies[antibody_name]

                            roi_name = list(antibody.rois.keys())[0]
                            slide_name = '_'.join(roi_name.split('_')[:-2])
                            
                            # write the ROI data, this is the most specific data
                            for roi_name in sorted(antibody.rois.keys()):
                                x = antibody.rois[roi_name].get_ao_data(measurement)
                                z = z_score(
                                    x, subregion_models[subregion_name]['model'])
                                write_row(
                                    roi_fps, slide_name, aid, bid,
                                    hemisphere_name, region_name, subregion_name, roi_name,
                                    antibody_name, measurement, x, z,
                                )
                            #
                            # end of ROIs loop

                            # write a row for the average over the ROIs in the subregion
                            x = subregion.collapse_ao(antibody_name, measurement)
                            z = z_score(
                                x, subregion_models[subregion_name]['model'])
                            write_row(
                                subregion_fps, slide_name, aid, bid,
                                hemisphere_name, region_name, subregion_name, 'AVG',
                                antibody_name, measurement, x, z,
                            )
                        #
                        # end of subregions loop

                        # write a row for the average over the subregions in the region
                        x = region.collapse_ao(antibody_name, measurement)
                        z = z_score(x, region_models[region_name]['model'])
                        write_row(
                            region_fps, slide_name, aid, bid,
                            hemisphere_name, region_name, 'AVG', 'AVG',
                            antibody_name, measurement, x, z,
                        )

                        # # special case: combines a24(a-c) subregions as a24 for aCING
                        # if region.name == 'aCING':
                        #     x = region.collapse_ao(antibody_name, measurement,
                        #                            subregions=['a24a', 'a24b', 'a24c'])
                        #     z = z_score(x, region_models[region_name]['model'])
                        #     write_row(
                        #         region_fps, slide_name, aid, bid,
                        #         hemisphere_name, region_name, 'a24abc', 'AVG',
                        #         antibody_name, measurement, x, z,
                        #     )
                    #
                    # end of regions loop

                    # write a row for the average over all regions in this hemisphere
                    x = hemisphere.collapse_ao(antibody_name, measurement)
                    z = z_score(x, hemisphere_models[hemisphere_name]['model'])
                    write_row(
                        hemisphere_fps, slide_name, aid, bid,
                        hemisphere_name, 'AVG', 'AVG', 'AVG',
                        antibody_name, measurement, x, z,
                    )
                #
                # end of hemispheres_loop

                # write a row for the average over all data in the patient
                x = patient.collapse_ao(antibody_name, measurement)
                z = z_score(x, patient_model)
                write_row(
                    patient_fps, slide_name, aid, bid,
                    'AVG', 'AVG', 'AVG', 'AVG',
                    antibody_name, measurement, x, z,
                )
            #
            # end of aids loop
        #
        # end of measurements loop
    #
    # end of antibodies loop

    # write the HC models to a spreadsheet as well
    for antibody_name in hc_models:
        for measurement in hc_models[antibody_name]:
            hemisphere_models = hc_models[antibody_name][measurement]
            for hemisphere_name in hemisphere_models:
                if hemisphere_name == 'model':
                    write_model_row(
                        models_fp,
                        'AVG', 'AVG', 'AVG', 'AVG',
                        antibody_name, measurement, hemisphere_models['model'],
                    )
                    continue
                region_models = hemisphere_models[hemisphere_name]
                for region_name in region_models:
                    if region_name == 'model':
                        write_model_row(
                            models_fp,
                            hemisphere_name, 'AVG', 'AVG', 'AVG',
                            antibody_name, measurement, region_models['model']
                        )
                        continue
                    subregion_models = region_models[region_name]
                    for subregion_name in subregion_models:
                        if subregion_name == 'model':
                            write_model_row(
                                models_fp,
                                hemisphere_name, region_name, 'AVG', 'AVG',
                                antibody_name, measurement, subregion_models['model'])
                            continue
                        write_model_row(
                            models_fp,
                            hemisphere_name, region_name, subregion_name, 'AVG',
                            antibody_name, measurement, subregion_models[subregion_name]['model']
                        )
#
# end of generate_spreadsheets
def write_model_row(fp, hemisphere_name, region_name, subregion_name, roi_name,
                    antibody_name, measurement, x):
    fp.write(
        WIDE_LINE % \
        ('HC_MODEL', 'HC_MODEL', 'HC_MODEL', hemisphere_name, region_name, subregion_name, roi_name,
         antibody_name, measurement+'_MU',
         *tuple(x[0]), *tuple([np.nan]*len(LAYER_NAMES))))
    fp.write(
        WIDE_LINE % \
        ('HC_MODEL', 'HC_MODEL', 'HC_MODEL', hemisphere_name, region_name, subregion_name, roi_name,
         antibody_name, measurement+'_SG',
         *tuple(x[1]), *tuple([np.nan]*len(LAYER_NAMES))))
#
# end of write_model_row

def write_row(fps, slide_name, aid, bid,
              hemisphere_name, region_name, subregion_name, roi_name,
              antibody_name, measurement, x, z):
    long_fp, wide_fp = fps
    for i in range(len(x)):
        long_fp.write(
            LONG_LINE % \
            (slide_name, aid, bid, hemisphere_name, region_name, subregion_name, roi_name,
             antibody_name, measurement, LAYER_NAMES[i], x[i], z[i]))
    wide_fp.write(
        WIDE_LINE % \
        (slide_name, aid, bid, hemisphere_name, region_name, subregion_name, roi_name, antibody_name, measurement,
         *tuple(x), *tuple(z)))
#
# end of write_row
    
def main(argv):

    # parse the command line
    parser = cmdl_parser(argv)
    args = parser.parse_args()

    # dataset of Patients containing hierarchical data
    patients, antibodies = collect_patients(args.idir, args.hc, args.hemi, args.exclude)
    
    # flatten the hierarchical data into a table
    generate_spreadsheets(args.odir, patients, antibodies)

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
        '-hc', type=str, default='',
        help="file containing a list of HC cases")
    parser.add_argument(
        '-hemi', type=str, default='',
        help='file containing a list of cases w/ hemisphere information'
    )
    parser.add_argument(
        '-exclude', type=str, default='',
        help='file containing slide names to exclude in the analysis'
    )
    return parser
#
# end of cmdl_parser

if __name__ == '__main__':
    main(sys.argv)
    
#
# end of file

