# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: Federico Barabas
"""
# Python2-3 compatibility, not tested
from __future__ import division, with_statement, print_function

import os

from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Python2-3 compatiblilty handling
try:
    from tkinter import Tk, filedialog

except ImportError:
    import tkFileDialog as filedialog
    from Tkinter import Tk

import h5py as hdf

from glob import glob
import beam_profile as bp

initialdir = 'Q:\\\\01_JointProjects\\STORM\\Switching\\data\\'
initialdir2 = '\\\\hell-fs\\STORM\\Switching\\data\\'
results_file = 'switching_results.hdf5'

# Cutoffs info handling
#
#dt = np.dtype([('path', 'S200'), ('cutoff', int)])
#cut = np.zeros(25, dtype=dt)
#cut[0] = (subdir + '/' + 'b760mW_0p1_uv0_100Hz_TIRF_000', 3000)
#ff = hdf.File(initialdir2 + 'switching_results.hdf5', 'r+')
#ff.create_dataset('/atto488/cutoffs',
#                  data=cut,
#                  maxshape=(None,))
#ff = hdf.File(initialdir2 + results_file, 'r+')
#prev = ff['alexa647/cutoffs'].value
#prev_size = ff['alexa647/cutoffs'].size
#ff['alexa647/cutoffs'].resize((prev_size + cut.size,))
#ff['alexa647/cutoffs'][prev_size:] = cut
#ff.close()

# Data type for the results
r_dtype = np.dtype([('date', int),
                    ('frame_rate', float),
                    ('n_frames', int),
                    ('cutoffs', (int, (20))),
                    ('frame_size', 'S10'),
                    ('intensity_ex', float),
                    ('intensity_405', float),
                    ('datasets', ('S3', (20))),
                    ('n_counts', (int, (20))),
                    ('inv_tau', float),
                    ('einv_tau', float),
                    ('hist_mean', float),
                    ('transitions', float),
                    ('etransitions', float),
                    ('path', 'S100')])


# CCD relative sensibility
ccd_dtype = np.dtype([('DU860', float),
                    ('DU897', float)])
ccd_sens = np.zeros(11, dtype=ccd_dtype)
ccd_sens['DU860'][10] = 1.56
ccd_sens['DU860'][5] = 1.22
ccd_sens['DU860'][3] = 1.17
ccd_sens['DU897'][10] = 1.17
ccd_sens['DU897'][5] = 1.04
ccd_sens['DU897'][3] = 1


def index_changes(x):
    return np.where(x[:-1] != x[1:])[0] + 1


def expo(x, A, inv_tau):
    return A * np.exp(- inv_tau * x)


def hyperbolic(x, A, B):
    return A * x / (1 + x/B)


def prop(x, A):
    return A * x


def linear(x, A, B):
    return A * x + B


def new_empty():
    """Method for creating a new empty results hdf5 file"""

    os.chdir(initialdir)

    splited_name = results_file.split(os.extsep)
    store_file_name = splited_name[0] + '_empty_new.' + splited_name[1]
    store_file = hdf.File(store_file_name, "w")

    # laser_calibration
    r_dtype = np.dtype([('date', int),
                        ('642_0', float),
                        ('642_linear', float),
                        ('642_quad', float),
                        ('405_0', float),
                        ('405_linear', float),
                        ('405_quad', float),
                        ('comment', 'S100')])
    calibration = np.zeros(6, dtype=r_dtype)
    store_file.create_dataset('laser_calibration',
                              data=calibration,
                              maxshape=(None,))

    # laser_calibration_units
    l_dtype = np.dtype([('laser', int),
                        ('units', 'S100')])
    cal_units = np.zeros(2, dtype=l_dtype)
    cal_units['laser'][0] = 405
    cal_units['laser'][1] = 642
    cal_units['units'][0] = r'labview scale to W/cm^2'
    cal_units['units'][1] = r'mW to kW/cm^2'
    store_file.create_dataset('laser_calibration_units',
                              data=cal_units,
                              maxshape=(None,))
    store_file.close()


def new_calibration(load_dir=initialdir, load_file=results_file):
    results = getresults(load_dir=load_dir, load_file=load_file)
    size = results['laser_calibration'].size
    results['laser_calibration'].resize((size + 1,))
    results.close()


def load_dir(initialdir=initialdir):

    from glob import glob

    # Get working folder from user
    try:

        root = Tk()
        dir_name = filedialog.askdirectory(parent=root,
                                           initialdir=initialdir,
                                           title='Please select a directory')
        root.destroy()

    except OSError:
        print("No folder selected!")

    # Get subdirectories of the chosen folder
    subdirs = [x[0] for x in os.walk(dir_name)]

    dir_names = []
    return_lists = []
    #for subdir in subdirs[1:]:
    for subdir in subdirs:

        dir_names.append(subdir)
        os.chdir(subdir)

        # Load list of files of the required type
        file_list = glob("*.sw_*")

        # Look for duplicates and put them in the same list
        return_list = []

        for file in file_list:
            matching = [s for s in file_list if file[0:file.find('.')-3] in s]
            return_list.append(matching)

        return_lists.append([list(t) for t in set(map(tuple, return_list))])

    return dir_names, return_lists


def getresults(load_dir=initialdir2, load_file=results_file):
    """Load results held in the hdf5 results file"""

    if os.path.isfile(load_dir + load_file):

        # Load data from HDF5 file
        return hdf.File(load_dir + load_file, "r+")

    else:
        print("File not found")

        return None


class Data:
    """Methods for analyzing the switching dynamics data"""

    def load(self, dir_name=None, file_name=None, initialdir=initialdir,
             bins=50, last=-1, fit_start=1):
        """Data loading
        file_name can be:
            ~) a string containing the name of the file to load
            ~) a list of strings, containing the names of the files that
            should be considered part of the same dataset"""

        # Total bins
        self.bins = bins

        # Number of bins excluded from the analysis
        self.fit_start = fit_start

        if dir_name is None:
            # File dialog
            root = Tk()
            file_name = filedialog.askopenfilenames(parent=root,
                                                    initialdir=initialdir,
                                                    title='Please select all '
                                                    'files that have to be '
                                                    'joined')
            root.destroy()

            # File attributes definitions
            file_name = file_name.split()
            self.dir_name = os.path.split(file_name[0])[0]
            self.file_name = [os.path.split(name)[1] for name in file_name]

        else:
            self.dir_name = dir_name
            self.file_name = file_name

        os.chdir(self.dir_name)
        self.subdir = os.path.split(self.dir_name)[1]
        self.date = int(self.subdir[0:6])

        # Joining measurements
        if type(self.file_name) is list:
            self.joined = True

        else:
            self.joined = False
            self.file_name = [self.file_name]

        # Paths and parameter extraction
        self.path = self.file_name
        print(self.file_name)
        file_name0 = self.file_name[0]
        self.nfiles = len(self.file_name)
        self.minipath = (self.subdir + r"/" + file_name0.split('.')[0][:-4])
        self.parameter = str(file_name0.split('.')[1].split('_')[1])
        self.power = int(file_name0.split('_')[0][1:-2])
        self.aotf = float(file_name0.split('_')[1].replace('p', '.'))

        uv_pos = file_name0.split('uv')[1]
        self.aotf405 = float(uv_pos[:uv_pos.find('_')].replace('p', '.'))
        if file_name0[0] == 'b':
            self.laser = '488'
            self.dye = 'atto488'
        elif file_name0[0] == 'r':
            self.laser = '642'
            self.dye = 'alexa647'
        elif file_name0[0] == 'g':
            self.laser = '532'
            self.dye = 'alexa532'

        # Information extraction from .inf file
        inf_name = file_name0.split('.')[0] + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
        self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:])
        self.n_frames = float(inf_data[29][inf_data[29].find('=') + 1:])
        self.frame_size = (inf_data[8][inf_data[8].find('=') + 1:] + 'x' +
                           inf_data[8][inf_data[8].find('=') + 1:])
        self.camera = (inf_data[10][inf_data[10].find('=') + 1:
                       inf_data[10].find('_', inf_data[10].find('='))])
        self.hs_speed = int((inf_data[11][inf_data[11].find('=') + 1:
                             inf_data[11].find('_', inf_data[11].find('='))]))

        # Data loading
        self.datasets = np.zeros((20), dtype='S3')
        self.n_counts = np.zeros((20), dtype=int)
        self.cutoffs = np.zeros((20), dtype=int)

        cutoff_hdf = hdf.File(initialdir2 + results_file, 'r')
        cuts = cutoff_hdf[self.dye]['cutoffs']
        is_cut = [co in self.subdir + '/' + file_name0 for co in cuts['path']]

        if self.parameter in ['offtimes', 'ontimes', 'photons']:
            dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4'),
                           ('timestamp', '<f4')])
            if np.sum(is_cut) > 0:
                cut = cuts['cutoff'][np.argmax(is_cut)]
                table = np.fromfile(file_name0, dtype=dt)
                self.table = table[table['timestamp'] < cut]
                self.cutoffs[0] = cut
            else:
                self.table = np.fromfile(file_name0, dtype=dt)

        else:
            dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])
            self.table = np.fromfile(file_name0, dtype=dt)

        self.datasets[0] = file_name0.split('.')[0][-3:]
        self.n_counts[0] = len(self.table)

        if self.nfiles > 1:

            for n_file in np.arange(1, self.nfiles):
                f_name = self.file_name[n_file]
                is_cut = [co in self.subdir + '/' + f_name
                          for co in cuts['path']]
                new_table = np.fromfile(self.file_name[n_file], dtype=dt)

                if np.sum(is_cut) > 0:
                    cut = cuts['cutoff'][np.argmax(is_cut)]
                    self.cutoffs[n_file] = cut
                    new_table = new_table[new_table['timestamp'] < cut]

                self.table = np.concatenate((self.table, new_table))
                self.n_counts[n_file] = len(new_table)
                self.datasets[n_file] = f_name.split('.')[0][-3:]

        cutoff_hdf.close()

        self.total_counts = sum(self.n_counts)

        # Histogram construction
        par_table = self.table[self.parameter]
        self.mean = np.mean(par_table)
        print('mean', self.mean)

        # Mean width cannot be less than 1 because we're making an histogram
        # of number of FRAMES
#        if self.parameter in ['ontimes']:
#            self.bin_width = min([max([round(self.mean / 10), 1]), 4])
#        else:
        self.bin_width = max([round(self.mean / 10), 1])

        self.threshold = fit_start * self.bin_width
        self.thr_par_table = par_table[par_table > self.threshold]
        self.thr_mean = np.mean(self.thr_par_table) - self.threshold
        print('threshold:', self.threshold)
        print('thr_mean', self.thr_mean)
        self.mean = self.thr_mean

#        self.bin_width = max([round(self.mean / 10), 1])

        self.hist, self.bin_edges = np.histogram(self.thr_par_table,
                                                 bins=bins,
                                                 normed=True,
                                                 range=(self.threshold,
                                                        bins * self.bin_width))
        self.bin_centres = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.fitted = False

        # Load data from HDF5 file
        if os.path.isfile(initialdir2 + results_file):

            store_file = hdf.File(initialdir2 + results_file, "r")

        else:
            print("HDF5 file not found")

        # Laser's intensity calibration
        calibrations = np.array([int(d)
                                for d in store_file['calibrations'].keys()])
        cal_date = str(calibrations[np.argmax(calibrations > self.date) - 1])
        calibration = store_file['calibrations'][cal_date][self.laser].value

        cal_row = ((calibration['set power [mW]'] == self.power) *
                   (calibration['aotf'] == self.aotf))

        if np.sum(cal_row) != 1:
            print(np.sum(cal_row))
            print(self.power, self.aotf, self.laser)
            print("Error in the calibration routine")

        int_index = np.argmax(cal_row)
        self.intensity_ex = calibration['intensity [kW/cm2]'][int_index]

        cal405 = store_file['calibrations'][cal_date]['405'].value
        int405_index = np.argmax(cal405['aotf'] == self.aotf405)
        self.intensity_405 = cal405['intensity [W/cm2]'][int405_index]

        # Old method
#        index = np.argmax(store_file['laser_calibration']['date'] > self.date)
#        p0_642 = store_file['laser_calibration']['642_0'][index - 1]
#        p1_642 = store_file['laser_calibration']['642_linear'][index - 1]
#        p2_642 = store_file['laser_calibration']['642_quad'][index - 1]
#        p0_405 = store_file['laser_calibration']['405_0'][index - 1]
#        p1_405 = store_file['laser_calibration']['405_linear'][index - 1]
#        p2_405 = store_file['laser_calibration']['405_quad'][index - 1]
#        self.intensity642 = (p0_642 +
#                             p1_642 * self.power642 +
#                             p2_642 * self.power642**2)
#        self.intensity405 = (p0_405 +
#                             p1_405 * self.power405 +
#                             p2_405 * self.power405**2)

        store_file.close()

        # transitions calculation
        mol_tab = self.table['molecules']
        splitted = np.split(mol_tab, index_changes(mol_tab))
        transitions = np.array([len(s) for s in splitted])
        self.transitions = transitions.mean()
        self.etransitions = transitions.std()

    def fit(self):
        """Histogram fitting"""

        # Educated guess to initialize the fit
        self.fit_guess = [self.hist[0], 1 / self.mean]

        # Error estimation from Poisson statistics
        sigma = np.sqrt(self.hist)
        if 0 in sigma:
            sigma = np.asarray([1 if x == 0 else x for x in sigma])

        # Curve fitting
        try:
            x_fit = self.bin_centres
            y_fit = self.hist
            self.fit_par, self.fit_var = curve_fit(expo, x_fit, y_fit,
                                                   p0=self.fit_guess,
                                                   sigma=sigma)
            self.fitted = True

            # Method definitions to make it more verbose
            self.amplitude = self.fit_par[0]
            self.inv_tau = self.fit_par[1]
            self.einv_tau = np.sqrt(self.fit_var[1, 1])
            if self.parameter in ['offtimes', 'ontimes']:
                self.inv_tau = self.inv_tau * self.frame_rate
                self.einv_tau = self.einv_tau * self.frame_rate
                self.mean = self.mean / self.frame_rate

            if self.parameter in ['photons', 'totalphotons']:
                ccd_factor = ccd_sens[self.camera][self.hs_speed]
                self.inv_tau = self.inv_tau / ccd_factor
                self.einv_tau = self.einv_tau / ccd_factor
                self.mean = self.mean * ccd_factor

        except RuntimeError:
            print("Fit didn't converge for", self.file_name)
            self.fitted = False
            self.amplitude = 0
            self.inv_tau = 0
            self.fitted = False

        self.results = np.array([(self.date,
                                  self.frame_rate,
                                  self.n_frames,
                                  self.cutoffs,
                                  self.frame_size,
                                  self.intensity_ex,
                                  self.intensity_405,
                                  self.datasets,
                                  self.n_counts,
                                  self.inv_tau,
                                  self.einv_tau,
                                  self.mean,
                                  self.transitions,
                                  self.etransitions,
                                  self.minipath)],
                                dtype=r_dtype)

    def plot(self):
        """Data plotting.
        If the data was fitted, the fitting function is plotted too."""

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.minipath)
        self.ax.bar(self.bin_centres, self.hist, self.bin_width, alpha=0.5,
                    label='_nolegend_')
        self.ax.set_xlabel(self.parameter)
        self.ax.grid(True)

        self.ax.set_xlim(0, self.bins * self.bin_width)
        self.ax.set_ylim(0, 1.1 * self.hist[1])
        self.ax.text(0.75 * self.bins * self.bin_width,
                     0.2 * self.ax.get_ylim()[1],
                     "Counts after 1st bin:\n" + str(len(self.thr_par_table)),
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white'))

        # Plot of the distribution with lambda = 1 / mean
        hist_est = expo(self.bin_centres, 1 / self.thr_mean, 1 / self.thr_mean)
        self.ax.plot(self.bin_centres, hist_est, color='b', lw=3,
                     label="Estimator distribution\n"
                     "y = (1 / mean) * exp(-(1 / mean) * x)\n"
                     "mean = {}".format(self.mean))

        # If the histogram was fit, then we plot also the fitting exponential
        if self.fitted:
            hist_fit = expo(self.bin_centres, *self.fit_par)
            self.ax.plot(self.bin_centres, hist_fit, color='r', lw=3,
                         label="Fitting distribution\n"
                         "y = A * exp(-inv_tau * x)\nA = {}\ninv_tau = "
                         "{}\ntau = {}".format(self.amplitude,
                                               self.inv_tau,
                                               1 / self.inv_tau))
            # Print filter indicators
            print("total_counts (200) =", self.total_counts)
            mean_pos = 1 / (self.inv_tau * self.bin_width)
            if self.parameter in ['offtimes', 'ontimes']:
                mean_pos = mean_pos * self.frame_rate
            print("mean_pos (1.2) =", mean_pos)
            print("hist_mean", self.mean)
            print("intensity", self.intensity_ex)

        self.ax.legend()
        plt.show()

    def save(self, store_name=os.path.join(initialdir2, results_file)):
        """Save in disk the results of the fitting of this instance of Data"""

        if self.fitted:

#            cwd = os.getcwd()
#            os.chdir(os.path.split(cwd)[0])

            # If file doesn't exist, it's created and the dataset is added
#            if not(os.path.isfile(store_name)):
#                store_file = hdf.File(store_name, "w")
#                store_file.create_dataset(self.parameter,
#                                          data=self.results,
#                                          maxshape=(None,))
#
#            else:

            store_file = hdf.File(store_name, "r+")

            # If file exists but the dataset doesn't
            dat_path = '/' + self.dye + '/' + self.parameter
            if not(dat_path in store_file):
                store_file.create_dataset(dat_path,
                                          data=self.results,
                                          maxshape=(None,))

            # If file exists and the dataset too,
            # we check if any of the new results were previously saved
            else:
                prev_size = store_file[dat_path].size
                prev_results = store_file[dat_path].value

                exists = np.any(prev_results == self.results)
                if exists:
                    print(self.results['path'], "was previously saved in",
                          store_name, r'/', self.parameter,
                          ". We won't save it again.")

                else:
                    store_file[dat_path].resize((prev_size + 1,))
                    store_file[dat_path][prev_size] = self.results

            store_file.close()
            print("Done!")

#            os.chdir(cwd)

        else:
            print("Can't save results, Data not fitted")


def analyze_file(from_bin=0):
    """Wrapper of processes for analyzing a file"""

    data = Data()
    data.load()
    data.fit(from_bin)
    data.plot()

    # Results printing
    print(data.parameter + " analyzed in " + data.dir_name)

    # If the plots are ok, Save results
    save = raw_input("Save results? (y/n) ")
    if save == 'y':
        data.save()


def save_folder(dye, parameter, new_results,
                store_name=initialdir2 + results_file):
    """Saves the fitting results of the analysis of all files in a folder."""

    cwd = os.getcwd()
    os.chdir(os.path.split(cwd)[0])
    print(store_name)

    # If file doesn't exist, it's created and the dataset is added
    if not(os.path.isfile(store_name)):
        store_file = hdf.File(store_name, "w")
        store_file.create_dataset(parameter,
                                  data=new_results,
                                  maxshape=(None,))

    else:
        store_file = hdf.File(store_name, "r+")

        dat_path = '/' + dye + '/' + parameter

        # If file exists but the dataset doesn't
        if not(dat_path in store_file):
            store_file.create_dataset(dat_path,
                                      data=new_results,
                                      maxshape=(None,))

        # If file exists and the dataset too,
        # we check if any of the new results were previously saved
        else:
            prev_size = store_file[dat_path].size
            prev_results = store_file[dat_path].value

            exists = np.array([np.any(prev_results == new_result)
                              for new_result in new_results],
                              dtype=bool)

            if np.any(exists):
                print(new_results[exists]['path'])
                print("was previously saved in", store_name, r'/', parameter,
                      ". We won't save it again.")
                new_results = new_results[np.logical_not(exists)]

            # Saving truly new data
            nfiles = new_results.size

            if nfiles > 0:
                store_file[dat_path].resize((prev_size + nfiles,))
                store_file[dat_path][prev_size:] = new_results

    store_file.close()
    print("Done!")

    os.chdir(cwd)


def analyze_folder(parameters, from_bin, quiet=False, save_all=False,
                   control=False, simulation=False, factors=None):

    # Saving thresholds
    if save_all:
        quiet = True
        min_counts = 10
        min_mean_pos = 0

    else:
        min_counts = 200
        min_mean_pos = 1.2

    # Conversion of parameters and from_bin to list and array
    if type(parameters) is not(list):
            parameters = [parameters]

    if from_bin == 0:
        from_bin = np.zeros((len(parameters), ))

    dir_names, files_lists = load_dir()

    if factors is None:

        epinames = glob(os.path.join(dir_names[0], 'profile_EPI*.dax'))
        tirfnames = glob(os.path.join(dir_names[0], 'profile_TIRF*.dax'))

        if len(epinames) > 0:
            tirf_factor, frame_factor, variance = bp.analyze_beam(epinames,
                                                                  tirfnames)

        else:
            print('EPI and TIRF measurements not provided')

    else:
        tirf_factor, frame_factor, variance = factors

    print('TIRF factor', tirf_factor)
    print('Frame factor', frame_factor)
    print('Intensity variance in frame', str(np.round(variance, 2))[:4], '%')

    for dir_name in dir_names:

        if len(files_lists[dir_names.index(dir_name)]) > 0:

            os.chdir(dir_name)

            for parameter in parameters:

                file_list = []
                for item in files_lists[dir_names.index(dir_name)]:
                    file_list.append([item_i for item_i in item if
                                     parameter in item_i])

                if control:
                    file_list[0].sort()
                    file_list = [file_list[0][i:i + 3]
                                 for i in range(0, len(file_list[0]), 3)]

                nfiles = len(file_list)

                # Creation of an instance of Data() for each file
                data_list = [Data() for file in file_list]

                # Structured array for the results of the folder
                folder_results = np.zeros(nfiles, dtype=r_dtype)

                # Process all files in a loop
                for i in range(nfiles):

                    # Procedures
                    data_list[i].load(dir_name, file_list[i],
                                      fit_start=from_bin[parameters.index(parameter)])
                    data_list[i].fit()

                    if not(simulation):
                        if data_list[i].fitted:
                            if save_all or quiet:
                                min_total_counts = data_list[i].total_counts
                                mean_pos = (1 /
                                            (data_list[i].inv_tau *
                                             data_list[i].bin_width))
                                if data_list[i].parameter in ['offtimes',
                                                              'ontimes']:
                                    mean_pos = (mean_pos *
                                                data_list[i].frame_rate)
                                if (min_total_counts > min_counts and
                                    mean_pos > min_mean_pos):
                                    folder_results[i] = data_list[i].results

                            # Conditions for automatic saving: min total_counts
                            # and min bin position of fitting mean
                            else:
                                data_list[i].plot()
                                save = raw_input("Save it? (y/n) ")
                                if save is 'y':
                                    folder_results[i] = data_list[i].results
                                    fac = tirf_factor * frame_factor
                                    tirff = tirf_factor
                                    folder_results[i]['intensity_ex'] *= fac
                                    folder_results[i]['intensity_405'] *= tirff

                # Results printing
                print(parameter + " analyzed in " + dir_name)

                # Get valid results and save them
                empty_line = np.zeros(1, dtype=r_dtype)
                folder_results = folder_results[np.where(folder_results !=
                                                         empty_line)]
                folder_results.sort(order=('date', 'intensity_ex'))

                if len(folder_results) > 0:
                    print('Saving', len(folder_results), 'out of', nfiles,
                          '...')
                    dye = data_list[0].dye
                    save_folder(dye, parameter, folder_results)

                else:
                    print("No data to save")


def duty_cycle(initialdir=initialdir, results_file=results_file):
    """Analyze the duty-cycle parameter in a selected folder"""

    dc_dtype = np.dtype([('date', int),
                        ('frame_rate', float),
                        ('n_frames', float),
                        ('frame_size', 'S10'),
                        ('intensity_ex', float),
                        ('intensity_405', float),
                        ('n_counts', (int, (10))),
                        ('dcycle_mean', float),
                        ('dcycle_std', float),
                        ('path', 'S100')])

    dir_names, files_lists = load_dir(initialdir=initialdir)

    for dir_name in dir_names:

        if len(files_lists[dir_names.index(dir_name)]) > 0:

            print("Analyzing folder", dir_name)

            onfiles = []
            offfiles = []

            for item in files_lists[dir_names.index(dir_name)]:
                onfiles.append([item_i for item_i in item if
                                item_i.endswith("_ontimes")])
                offfiles.append([item_i for item_i in item if
                                item_i.endswith("_offtimes")])

            nfiles = len(onfiles)

            # Creation of an instance of Data() for each file
            ondata = [Data() for file in onfiles]
            offdata = [Data() for file in offfiles]

            # Structured array for the results of the folder
            folder_results = np.zeros(nfiles, dtype=dc_dtype)

            # Load the files and store metadata
            for i in np.arange(nfiles):

                ondata[i].load(dir_name, onfiles[i])
                offdata[i].load(dir_name, offfiles[i])
                folder_results[i] = np.array([(offdata[i].date,
                                             offdata[i].frame_rate,
                                             offdata[i].n_frames,
                                             offdata[i].frame_size,
                                             offdata[i].power642,
                                             offdata[i].intensity642,
                                             offdata[i].power405,
                                             offdata[i].intensity405,
                                             offdata[i].n_counts,
                                             0,
                                             0,
                                             offdata[i].minipath)],
                                             dtype=dc_dtype)

            # Sort the data
            ondata = [np.sort(ondata_i.table, order=['molecules'])
                      for ondata_i in ondata]
            offdata = [np.sort(offdata_i.table, order=['molecules'])
                       for offdata_i in offdata]

            # Split data according to the molecule number
            on_indexes = [np.unique(ondata_i['molecules'], True)[1]
                          for ondata_i in ondata]
            off_indexes = [np.unique(offdata_i['molecules'], True)[1]
                           for offdata_i in offdata]
            ontimesxmol = [np.split(ondata[j]['ontimes'],
                                    on_indexes[j])[1:]
                           for j in np.arange(len(ondata))]
            offtimesxmol = [np.split(offdata[j]['offtimes'],
                                     off_indexes[j])[1:]
                            for j in np.arange(len(offdata))]

            # Duty cycle calculation
            s_ontimes = np.array([np.array([np.sum(ontimes)
                                  for ontimes in ontimesxmol_i])
                                  for ontimesxmol_i in ontimesxmol])
            s_offtimes = np.array([np.array([np.sum(offtimes)
                                   for offtimes in offtimesxmol_i])
                                   for offtimesxmol_i in offtimesxmol])

            dutys = np.array([s_ontimes[j] / (s_ontimes[j] + s_offtimes[j])
                              for j in np.arange(len(s_ontimes))])

            duty_mean = np.array([np.mean(duty) for duty in dutys])
            duty_std = np.array([np.std(duty) for duty in dutys])

            # Save results
            folder_results['dcycle_mean'] = duty_mean
            folder_results['dcycle_std'] = duty_std
            save_folder('duty_cycle', folder_results)


def load_results(dye, parameter, load_dir=initialdir,
                 results_file=results_file, mean=False, fit=None, fit_end=None,
                 interval=None, dates=[0, 990000], join=False,
                 discriminate=False):
    """Plot results held in results_vs_power.hdf5 file"""

    store_name = results_file
    os.chdir(load_dir)

    if os.path.isfile(store_name):

        # Load data from HDF5 file
        infile = hdf.File(store_name, "r")
        results = infile[dye][parameter].value
#        calibration = infile['laser_calibration'].value
        infile.close()

        # Define subset of data
        results = np.sort(results, order=['date'])
        results = results[((results['date'] >= dates[0]) *
                           (results['date'] <= dates[1]))]

        # Plot
        fig, ax = plt.subplots()
        x_data = results['intensity_ex']
        if parameter is 'duty_cycle':
            y_data = results['dcycle_mean']
            ey_data = results['dcycle_std']
        else:
            if mean:
                y_data = 1 / results['hist_mean']
            else:
                y_data = results['inv_tau']
                ey_data = results['einv_tau']

        dates = results['date']
        dates = np.unique(dates, return_index=True, return_inverse=True)[2]

        ax.set_xlabel("Intensity [kW/cm^2]")
        ax.set_xlim(0, int(ceil(x_data.max() / 10 + 1)) * 10)
#        ax.set_xlim(0, 170)

        if parameter is "ontimes":

            plt.errorbar(x_data, y_data, yerr=ey_data, fmt='o') #, facecolors='none', edgecolors='b')
                                        # c=dates,

            if interval is None:
                ax.set_ylim(0, int(ceil(y_data.max() / 100.0)) * 100)

            else:
                ax.set_ylim(interval[0], interval[1])

            ax.set_ylabel("Off rate [s^-1]")

        elif parameter is "offtimes":

#            if discriminate:
#                indices = [np.where(results['date'] >= cal_date)[0][0]
#                           for cal_date in calibration['date']]
#                x_r = np.array(np.split(x_data, indices))[1:]
#                y_r = np.array(np.split(y_data, indices))[1:]
#
#                colors = ['w', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
#
#                for i in np.arange(x_r.size):
#                    if 'TIRF' in calibration['comment'][i]:
#                        plt.scatter(x_r[i], y_r[i], facecolors=colors[i])
#                    # , facecolors='none', edgecolors='b')

#            elif not(join):
            plt.scatter(x_data, y_data, facecolors='none', edgecolors='b')

            ax.set_ylabel("On rate [s^-1]")
#            ax.set_xlim(0, 30)

            if interval is not None:
                ax.set_ylim(interval[0], interval[1])

        elif parameter is "duty_cycle":
            if not(join):
                plt.errorbar(x_data, y_data, ey_data, fmt='o')
            ax.set_ylabel("Duty cycle")
            ax.set_ylim(0, 0.01)


        else:

            if mean:
                y_data = results['hist_mean']

            else:
                y_data = 1 / results['inv_tau']

            if not(join):
                plt.scatter(x_data, y_data, facecolors='none', edgecolors='b')
                                # c=dates

            if interval is not None:
                ax.set_ylim(interval[0], interval[1])

            ax.set_ylabel(parameter)

        # Curve fitting
        if fit_end is not None:
            end = np.argmax(x_data > fit_end)
            x_data = x_data[:end]
            y_data = y_data[:end]

        if fit is 'hyperbolic':

            init_slope = 10
            guess = [init_slope, y_data.max() / init_slope]

            fit_par, fit_var = curve_fit(hyperbolic, x_data, y_data,
                                         p0=guess)

            # Get the sigma of parameters from covariance matrix
            fit_sigma = np.sqrt(fit_var.diagonal())

            # Fitting curve plotting
            x_plot = np.array(x_data)
            x_plot.sort()
            fit_func = hyperbolic(x_plot, *fit_par)
            ax.plot(x_plot, fit_func, color='r', lw=3,
                    label="y = A * x / (1 + x/B)\nA = {} pm {} \nB = {} pm {}"
                    .format(np.round(fit_par[0], 1),
                            np.round(fit_sigma[0], 1),
                            np.round(fit_par[1], 1),
                            np.round(fit_sigma[1], 1)))
            ax.legend(loc=4)

        elif fit is 'prop':

            guess = [y_data.max() / x_data.max()]
            fit_par, fit_var = curve_fit(prop, x_data, y_data, p0=guess)
            fit_sigma = np.sqrt(fit_var[0])

            # Fitting curve plotting
            fit_func = prop(x_data, *fit_par)
            ax.plot(x_data, fit_func, color='r', lw=3,
                    label="y = A * x\nA = {} pm {}"
                    .format(np.round(fit_par[0], 5),
                            np.round(fit_sigma[0], 5)))
            ax.legend(loc=4)

        elif fit is 'linear':

            guess = [y_data.max() / x_data.max(), y_data.min()]
            fit_par, fit_var = curve_fit(linear, x_data, y_data, p0=guess)
            fit_sigma = np.sqrt(fit_var.diagonal())

            # Fitting curve plotting
            fit_func = linear(x_data, *fit_par)
            ax.plot(x_data, fit_func, color='r', lw=3,
                    label="y = A * x + B\nA = {} pm {}\nB = {} pm {}"
                    .format(np.round(fit_par[0], 3),
                            np.round(fit_sigma[0], 3),
                            np.round(fit_par[1], 2),
                            np.round(fit_sigma[1], 2)))
            ax.legend(loc=4)

        if join:

            # Sort the data
            y_s = y_data[np.argsort(x_data)]
            x_s = np.sort(x_data)

            if parameter in ['duty_cycle', 'offtimes']:

                x_u, index = np.unique(x_data, return_inverse=True)
                y_u = np.zeros(x_u.size)
                ey_u = np.zeros(x_u.size)
                for i in np.arange(np.unique(x_data).size):
                    data = y_data[np.where(index == i)]
                    if data.size > 1:
                        y_u[i] = data.mean()
                        ey_u[i] = data.std()
                    else:
                        y_u[i] = data[0]
#                        ey_u[i] = ey_data[np.where(index == i)][0]

                plt.errorbar(x_u, y_u, ey_u, fmt='o')

            else:

                # Group the data every 5 kW/cm2 steps
                last = x_s[-1]
                indices = [np.where(x_s > i)[0][0] for i in np.arange(0,
                                                                      last,
                                                                      5)]
                x_r = np.split(x_s, indices)

                # New x, y coordinates are the means of each group
                x_u = np.array([x_ri.mean() for x_ri in x_r if x_ri.size > 0])
                ex_u = np.array([x_ri.std() for x_ri in x_r if x_ri.size > 0])

                y_u = np.array([y_s[indices[i]:indices[i + 1]].mean()
                               for i in np.arange(x_u.size)])
                ey_u = np.array([y_s[indices[i]:indices[i + 1]].std()
                                for i in np.arange(x_u.size)])

                plt.errorbar(x_u, y_u, yerr=ey_u, xerr=ex_u, fmt='o')

        ax.grid(True)
        plt.show()

    else:

        print("File " + store_name + " not found in " + os.getcwd())


if __name__ == "__main__":

#    %load_ext autoreload
#    %autoreload 2

    import sys

    repos = 'P:\\Private\\repos'
    sys.path.append(repos)

    parameter = ['offtimes', 'ontimes', 'photons', 'totalphotons',
                 'transitions']
    first_bin = [1, 3, 3, 3, 3]

    import switching_analysis.analysis as sw

#    sw.analyze_folder(parameter, first_bin, quiet=True, save_all=True)

    sw.analyze_folder(['offtimes', 'ontimes'], first_bin)

#    sw.

#    results = sw.getresults(load_file='results_vs_power.hdf5')

#    sw.load_results(parameter[0])
