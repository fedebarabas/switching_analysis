# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: Federico Barabas
"""

import os

from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tkinter import Tk, filedialog
import h5py as hdf

# Data type for the results
r_dtype = np.dtype([('date', int),
                    ('frame_rate', float),
                    ('n_frames', float),
                    ('frame_size', 'S10'),
                    ('power_642', int),
                    ('intensity_642', float),
                    ('power_405', int),
                    ('intensity_405', float),
                    ('n_counts', (int, (10))),
                    ('inv_tau', float),
                    ('path', 'S100')])

initialdir = '\\\\hell-fs\\STORM\\Switching\\data\\'
results_file = 'results_vs_power.hdf5'

def expo(x, A, inv_tau):
    return A * np.exp(- inv_tau * x)

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

    #os.chdir(load_dir)

    if os.path.isfile(load_dir + load_file):

        # Load data from HDF5 file
        return hdf.File(load_dir + load_file, "r+")

    else:
        print("File not found")

        return None

class Data:
    """Methods for analyzing the switching dynamics data"""

    #def load(self, dir_name, file_name, bins=50):
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
        self.path = self.file_name
        self.nfiles = len(self.file_name)
        self.minipath = (self.subdir + r"/" +
                         self.file_name[0][0:self.file_name[0].find('.') - 4])
        self.parameter = self.file_name[0][self.file_name[0].rindex('_')
                                                    + 1:len(self.file_name[0])]
        self.power = int(self.file_name[0][1:self.file_name[0].index('_') - 2])

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

        # Information extraction from .inf file
        inf_name = self.file_name[0][:self.file_name[0].find('.')] + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
        self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:-1])
        self.n_frames = float(inf_data[29][inf_data[29].find('=') + 1:-1])
        self.frame_size = (inf_data[8][inf_data[8].find('=') + 1:-1] + 'x' +
                                    inf_data[8][inf_data[8].find('=') + 1:-1])


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

        store_file = getresults()

        # Filling the new rows
        intercept = store_file['calibration']['642_y_intercept'][-1]
        slope = store_file['calibration']['642_slope'][-1]
        area = store_file['area'].value
        self.intensity = (intercept + self.power * slope)/(1000 * 100 * area)

        n_counts_tmp = np.zeros((10), dtype=int)
        n_counts_tmp[0:len(self.n_counts)] = self.n_counts
        self.n_counts = n_counts_tmp
        self.results = np.array([(self.date,
                                  self.frame_rate,
                                  self.n_frames,
                                  self.frame_size,
                                  self.power,
                                  self.intensity,
                                  0,
                                  0,
                                  self.n_counts,
                                  self.inv_tau,
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

        plt.show()

    def save(self, store_name=results_file):
        """Save in disk the results of the fitting of this instance of Data"""

        if self.fitted:

            print("Saving...")
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

    print("Saving...")
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

def analyze_folder(parameters, from_bin=0, quiet=False, recursive=True):

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
                    if not(quiet):
                        data_list[i].plot()
                        save = input("Save it? (y/n) ")
                        if save=='y':
                            folder_results[i] = data_list[i].results

                    elif data_list[i].total_counts > 200:
                        folder_results[i] = data_list[i].results

                # Results printing
                print(parameter + " analyzed in " + dir_name)

                # Get valid results and save them
                empty_line = np.zeros(1, dtype=r_dtype)
                folder_results = folder_results[np.where(folder_results != empty_line)]

                print('Saving', len(folder_results), 'out of', nfiles)
                save_folder(parameter, folder_results)


def load_results(parameter, inv=True, load_dir=initialdir,
                                                 results_file=results_file):
    """Plot results held in results_vs_power.hdf5 file"""

    store_name = results_file
    os.chdir(load_dir)

    if os.path.isfile(store_name):

        # Load data from HDF5 file
        infile = hdf.File(store_name, "r")
        results = infile[parameter].value
        infile.close()

        # Plot
        fig, ax = plt.subplots()

        if inv:
            plt.scatter(results['intensity_642'], results['inv_tau'])
            ax.set_ylim(0, int(ceil(results['inv_tau'].max() / 100.0)) * 100)
            if parameter=="ontimes":
                ax.set_ylabel("Off rate [s^-1]")
            else:
                ax.set_ylabel("On rate [s^-1]")

        else:
            plt.scatter(results['intensity_642'], 1 / results['inv_tau'])
            ax.set_ylabel(parameter)

        ax.set_xlabel("Intensity [kW/cm^2]")
        ax.set_xlim(0, ceil(results['intensity_642'].max()))

        ax.grid(True)
        plt.show()

    else:

        print("File " + store_name + " not found in " + os.getcwd())

    ### TODO: put units to work

if __name__ == "__main__":

    parameter = ['ontimes']
    first_bin = [3, 3]

    import switching_analysis.analysis as sw

    import imp
    imp.reload(sw)

#    dir_name, return_list = sw.load_dir()

    sw.analyze_folder(parameter, first_bin)

    results = sw.getresults()

    sw.load_results(parameter[0])