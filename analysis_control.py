# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: Federico Barabas
"""
# Python2-3 compatibility, not tested
from __future__ import division, with_statement, print_function, absolute_import, unicode_literals

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

#initialdir = 'Q:\\\\01_JointProjects\\STORM\\Switching\\data\\'
initialdir = '\\\\hell-fs\\STORM\\Switching\\data\\'
results_file = 'results_control.hdf5'

# Data type for the results
#r_dtype = np.dtype([('date', int),
r_dtype = np.dtype([('date', float),
                    ('edate', float),
                    ('frame_rate', float),
                    ('n_frames', float),
                    ('frame_size', 'S10'),
                    ('power_642', int),
                    ('intensity_642', float),
                    ('power_405', int),
                    ('intensity_405', float),
                    ('n_counts', (int, (10))),
#                    ('n_counts', (int, (40))),
                    ('inv_tau', float),
                    ('hist_mean', float),
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

def expo(x, A, inv_tau):
    return A * np.exp(- inv_tau * x)

def hyperbolic(x, A, B):
    return A * x / (1 + x/B)

def linear(x, A):
    return A * x

def new_empty():
    """Method for creating a new empty results hdf5 file"""

    os.chdir(initialdir)

    store_file = hdf.File("results_vs_power_empty_new.hdf5", "w")

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
    results = getresults(load_file = load_dir + load_file)
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

        os.chdir(dir_name)

    except OSError:
        print("No folder selected!")

    # Get subdirectories of the chosen folder
    subdirs = [x[0] for x in os.walk(os.getcwd())]

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

    os.chdir(subdirs[0])

    return dir_names, return_lists

def getresults(load_dir=initialdir, load_file=results_file):
    """Load results held in results_vs_power.hdf5 file"""

    if os.path.isfile(load_dir + load_file):

        # Load data from HDF5 file
        return hdf.File(load_dir + load_file, "r+")

    else:
        print("File", load_file, "not found in", load_dir)

        return None

class Data:
    """Methods for analyzing the switching dynamics data"""

    def load(self, dir_name=None, file_name=None, initialdir=initialdir,
             bins=50):
        """Data loading
        file_name can be:
            ~) a string containing the name of the file to load
            ~) a list of strings, containing the names of the files that
            should be considered part of the same dataset"""

        self.bins = bins

        if dir_name==None:
            # File dialog
            root = Tk()
            file_name = filedialog.askopenfilename(parent=root,
                       initialdir=initialdir,
                       title='Please select all files that have to be joined')
            root.destroy()

            # File attributes definitions
            self.dir_name = os.path.split(file_name)[0]
            self.file_name = os.path.split(file_name)[1]

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

        dot_p = self.file_name[0].find('.')

        self.path = self.file_name
        self.nfiles = len(self.file_name)
        self.minipath = (self.subdir + r"/" +
                         self.file_name[0][0:dot_p - 4])
        self.parameter = self.file_name[0][self.file_name[0].rindex('_')
                                                    + 1:len(self.file_name[0])]
        self.power = int(self.file_name[0][1:self.file_name[0].index('_') - 2])

        # Measurement date&time
        dax_files = [name[:][:dot_p] + '.dax' for name in file_name]
        times = np.array([os.path.getctime(dax) for dax in dax_files])
        self.time = times.mean()
        self.etime = times.std()

        # Information extraction from .inf file
        inf_name = self.file_name[0][:dot_p] + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
        self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:-1])
        self.n_frames = float(inf_data[29][inf_data[29].find('=') + 1:-1])
        self.frame_size = (inf_data[8][inf_data[8].find('=') + 1:-1] + 'x' +
                           inf_data[8][inf_data[8].find('=') + 1:-1])
        self.camera = (inf_data[10][inf_data[10].find('=') + 1:
                       inf_data[10].find('_', inf_data[10].find('='))])
        self.hs_speed = int((inf_data[11][inf_data[11].find('=') + 1:
                             inf_data[11].find('_', inf_data[11].find('='))]))

        # Data loading
        self.n_counts = []
        dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])
        self.table = np.fromfile(self.file_name[0], dtype=dt)
        self.n_counts.append((len(self.table)))

        if self.nfiles > 1:
            for file in self.file_name[1:]:
                new_table = np.fromfile(file, dtype=dt)
                self.table = np.concatenate((self.table, new_table))
                self.n_counts.append((len(new_table)))

        self.total_counts = sum(self.n_counts)

        # Histogram construction
        self.mean = np.mean(self.table[self.parameter])

        # Mean width cannot be less than 1 because we're making an histogram
        # of number of FRAMES
        self.bin_width = max([round(self.mean / 12), 1])
        self.hist, bin_edges = np.histogram(self.table[self.parameter],
                                            bins=bins,
                                            range=(0, bins * self.bin_width))
        self.bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.fitted = False

    def fit(self, fit_start=0):
        """Histogram fitting"""

        self.fit_start = fit_start

        # Educated guess to initialize the fit
        self.fit_guess = [self.hist[0], 1 / self.mean]

        # Error estimation from Poisson statistics
        sigma = np.sqrt(self.hist[self.fit_start:])
        if 0 in sigma:
            sigma = np.asarray([1 if x == 0 else x for x in sigma])

        # Curve fitting
        try:
            self.fit_par, self.fit_var = curve_fit(expo,
                                               self.bin_centres[self.fit_start:],
                                               self.hist[self.fit_start:],
                                               p0=self.fit_guess,
                                               sigma=sigma)
            self.fitted = True

            # Method definitions to make it more verbose
            self.amplitude = self.fit_par[0]
            self.inv_tau = self.fit_par[1]
            if self.parameter in ['offtimes', 'ontimes']:
                self.inv_tau = self.inv_tau * self.frame_rate
                self.mean = self.mean / self.frame_rate

        except RuntimeError:
            print("Fit didn't converge for", self.file_name)
            self.fitted = False
            self.amplitude = 0
            self.inv_tau = 0

        store_file = getresults()

        # Filling the new rows
        index = np.argmax(store_file['laser_calibration']['date'] > self.date)
        zero = store_file['laser_calibration']['642_0'][index - 1]
        linear = store_file['laser_calibration']['642_linear'][index - 1]
        quad = store_file['laser_calibration']['642_quad'][index - 1]
        self.intensity = zero + self.power * linear + self.power**2 * quad

        if self.parameter in ['photons', 'totalphotons']:
            ccd_factor = ccd_sens[self.camera][self.hs_speed]
            self.inv_tau = self.inv_tau / ccd_factor
            self.mean = self.mean * ccd_factor

        n_counts_tmp = np.zeros((10), dtype=int)
        n_counts_tmp[0:len(self.n_counts)] = self.n_counts
        # parche horrendo, no quiero volver a empezar
        self.n_counts = n_counts_tmp[0:np.min([n_counts_tmp.size, 10])]
        self.results = np.array([(self.time,
                                  self.etime,
                                  self.frame_rate,
                                  self.n_frames,
                                  self.frame_size,
                                  self.power,
                                  self.intensity,
                                  0,
                                  0,
                                  self.n_counts,
                                  self.inv_tau,
                                  self.mean,
                                  self.file_name[0])],
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
        self.ax.text(0.75 * self.bins * self.bin_width,
                     0.2 * self.ax.get_ylim()[1],
                     "Number of counts:\n" + str(self.table.size),
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white'))
        self.ax.set_xlim(0, self.bins * self.bin_width)

        # If the histogram was fit, then we plot also the fitting exponential
        if self.fitted:
            hist_fit = expo(self.bin_centres, *self.fit_par)
            self.ax.plot(self.bin_centres[self.fit_start:],
                     hist_fit[self.fit_start:],
                     color='r', lw=3,
                     label="y = A * exp(-inv_tau * x)\nA = {}\ninv_tau = {}\n"
                             "tau = {}\npower = {}"
                     .format(int(self.amplitude), self.inv_tau,
                             1 / self.inv_tau, self.power))
            self.ax.legend()

            # Print filter indicators
            print("total_counts (200) =", self.total_counts)
            mean_pos = 1 / (self.inv_tau * self.bin_width)
            if self.parameter in ['offtimes', 'ontimes']:
                mean_pos = mean_pos * self.frame_rate
            print("mean_pos (1.2) =", mean_pos)
            print("hist_mean", self.mean)
            print("hist_mean sobre fr", self.mean / self.frame_rate)

        plt.show()

    def save(self, store_name=results_file):
        """Save in disk the results of the fitting of this instance of Data"""

        if self.fitted:

            cwd = os.getcwd()
            os.chdir(os.path.split(cwd)[0])


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
            if not(self.parameter in store_file):
                store_file.create_dataset(self.parameter,
                                          data=self.results,
                                          maxshape=(None,))

            # If file exists and the dataset too,
            # we check if any of the new results were previously saved
            else:
                prev_size = store_file[self.parameter].size
                prev_results = store_file[self.parameter].value

                exists = np.any(prev_results==self.results)
                if exists:
                    print(self.results['path'], "was previously saved in",
                          store_name, r'/', self.parameter,
                          ". We won't save it again.")

                else:
                    store_file[self.parameter].resize((prev_size + 1,))
                    store_file[self.parameter][prev_size] = self.results

            store_file.close()
            print("Done!")

            os.chdir(cwd)

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
    save = input("Save results? (y/n) ")
    if save == 'y':
        data.save()


def save_folder(parameter, new_results, store_name=results_file):
    """Saves the fitting results of the analysis of all files in a folder."""

    cwd = os.getcwd()
    os.chdir(os.path.split(cwd)[0])

    # If file doesn't exist, it's created and the dataset is added
    if not(os.path.isfile(store_name)):
        store_file = hdf.File(store_name, "w")
        store_file.create_dataset(parameter, data=new_results, maxshape=(None,))

    else:
        store_file = hdf.File(store_name, "r+")

        # If file exists but the dataset doesn't
        if not(parameter in store_file):
            store_file.create_dataset(parameter,
                                      data=new_results,
                                      maxshape=(None,))

        # If file exists and the dataset too,
        # we check if any of the new results were previously saved
        else:
            prev_size = store_file[parameter].size
            prev_results = store_file[parameter].value

            exists = np.array([np.any(prev_results==new_result) for new_result in new_results],
                              dtype=bool)

            if np.any(exists):
                print(new_results[exists]['path'])
                print("was previously saved in", store_name, r'/', parameter,
                      ". We won't save it again.")
                new_results = new_results[np.logical_not(exists)]

            # Saving truly new data
            nfiles = new_results.size

            if nfiles > 0:
                store_file[parameter].resize((prev_size + nfiles,))
                store_file[parameter][prev_size:] = new_results

    store_file.close()
    print("Done!")

    os.chdir(cwd)

def analyze_folder(parameters, from_bin, quiet=False, save_all=False,
                   control=True, simulation=False):

    # Saving thresholds
    if save_all:
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

    for dir_name in dir_names:

        if len(files_lists[dir_names.index(dir_name)]) > 0:

            os.chdir(dir_name)

            for parameter in parameters:

                file_list = []
                for item in files_lists[dir_names.index(dir_name)]:
                    file_list.append([item_i for item_i in item if
                                            item_i.endswith("_" + parameter)])


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
                    data_list[i].load(dir_name, file_list[i])
                    data_list[i].fit(from_bin[parameters.index(parameter)])

                    if not(simulation):
                        if not(quiet) or not(save_all):
                            data_list[i].plot()
                            save = input("Save it? (y/n) ")
                            if save=='y':
                                folder_results[i] = data_list[i].results

                        # Conditions for automatic saving: min total_counts and
                        # min bin position of fitting mean
                        else:
                            min_total_counts = data_list[i].total_counts
                            mean_pos = (1 /
                                (data_list[i].inv_tau * data_list[i].bin_width))
                            if data_list[i].parameter in ['offtimes', 'ontimes']:
                                mean_pos = mean_pos * data_list[i].frame_rate
                            if (min_total_counts > min_counts and
                                mean_pos > min_mean_pos):
                                folder_results[i] = data_list[i].results

                # Results printing
                print(parameter + " analyzed in " + dir_name)

                # Get valid results and save them
                empty_line = np.zeros(1, dtype=r_dtype)
                folder_results = folder_results[np.where(folder_results != empty_line)]
                folder_results.sort(order=('date', 'power_642'))

                if len(folder_results) > 0:
                    print('Saving', len(folder_results), 'out of', nfiles, '...')
                    save_folder(parameter, folder_results)

                else:
                    print("No data to save")

def load_results(parameter, load_dir=initialdir, results_file=results_file,
                 mean=False, fit=None, fit_end=None, interval=None,
                 dates=[0, 990000], join=False, control=True):
    """Plot results held in results_vs_power.hdf5 file"""

    store_name = results_file
    os.chdir(load_dir)

    if os.path.isfile(store_name):

        # Load data from HDF5 file
        infile = hdf.File(store_name, "r")
        results = infile[parameter].value
        infile.close()

        # Define subset of data
#        results = results[((results['date'] >= dates[0])*
#                           (results['date'] <= dates[1]))]

        # Plot
        fig, ax = plt.subplots()

        if not(control):

            x_data = results['intensity_642']

            dates = results['date']
            dates = np.unique(dates, return_index=True, return_inverse=True)[2]

            ax.set_xlabel("Intensity [kW/cm^2]")
            ax.set_xlim(0, int(ceil(x_data.max() / 10 + 1)) * 10)

        else:

            # Here we're assuming the data is already sorted by timestamp
            # Converting to minutes
            x_data = (results['date'] - results['date'][0]) / 60
            ex_data = results['edate'] / 60


        if mean:
            y_data = 1 / results['hist_mean']
        else:
            y_data = results['inv_tau']

        if parameter=="ontimes":

            plt.errorbar(x_data, y_data, xerr=ex_data, fmt='o')
#                         facecolors='none',
#                         edgecolors='b')
                                        # c=dates,

#            if interval == None:
#                ax.set_ylim(0, int(ceil(y_data.max() / 100.0)) * 100)
#
#            else:
#                ax.set_ylim(interval[0], interval[1])

            ax.set_ylabel("Off rate [s^-1]")

        elif parameter=="offtimes":
            plt.errorbar(x_data, y_data, xerr=ex_data, fmt='o')
#                         facecolors='none',
#                         edgecolors='b')
            ax.set_ylabel("On rate [s^-1]")

        else:

            if mean:
                y_data = results['hist_mean']

            else:
                y_data = 1 / results['inv_tau']

            if not(join):
                plt.errorbar(x_data, y_data, xerr=ex_data, fmt='o')
                                #  , edgecolors='b', facecolors='none', c=dates

            if interval != None:
                ax.set_ylim(interval[0], interval[1])

            ax.set_ylabel(parameter)

        # Curve fitting
        if fit_end != None:
            end = np.argmax(x_data > fit_end)
            x_data = x_data[:end]
            y_data = y_data[:end]

        if fit=='hyperbolic':

            init_slope = 10
            guess = [init_slope, y_data.max() / init_slope ]

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

        elif fit=='linear':

            guess = [y_data.max() / x_data.max()]
            fit_par, fit_var = curve_fit(linear, x_data, y_data, p0=guess)
            fit_sigma = np.sqrt(fit_var[0])

            # Fitting curve plotting
            fit_func = linear(x_data, *fit_par)
            ax.plot(x_data, fit_func, color='r', lw=3,
            label="y = A * x\nA = {} pm {}"
            .format(np.round(fit_par[0], 1),
                    np.round(fit_sigma[0], 1)))
            ax.legend(loc=4)

        if join:
            y_s = y_data[np.argsort(x_data)]
            x_s = np.sort(x_data)
            last = x_s[-1]
            indices = [np.where(x_s > i)[0][0] for i in np.arange(0, last, 5)]
            x_r = np.split(x_s, indices)

#            x_u, index = np.unique(x_data, return_inverse=True)
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

    import sys, os

    repos = 'P:\\Private\\repos'
#    repos = r'/home/federico/Documents/nanofisica/python'
    sys.path.append(repos)

    parameter = ['ontimes', 'photons', 'totalphotons', 'transitions']
    first_bin = [3, 3, 3, 3]

    import switching_analysis.analysis_control as swc

#    swc.analyze_folder(parameter, first_bin, quiet=True, save_all=True)

    swc.analyze_folder(parameter[0], first_bin)

#    results = sw.getresults(load_file='results_vs_power.hdf5')

    swc.load_results(parameter[0])