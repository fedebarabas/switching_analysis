# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: Federico Barabas
"""

import os
import sys
import numpy as np
import h5py as hdf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def expo(x, A, inv_tau):
    return A * np.exp(- inv_tau * x)

def load_dir(initialdir='\\\\hell-fs\\STORM\\Switching\\'):

    from tkinter import Tk, filedialog
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

    # Load list of files of the required type
    file_list = glob("*.sw_*")

    # Look for duplicates and put them in the same list
    return_list = []

    for file in file_list:
        matching = [s for s in file_list if file[0:file.find('.')-3] in s]
        return_list.append(matching)

    return_list =  [list(t) for t in set(map(tuple, return_list))]

    return dir_name, return_list

class Data:
    """Methods for analyzing the switching dynamics data"""

    def load(self, dir_name, file_name, bins=50):
        """Data loading
        file_name can be:
            ~) a string containing the name of the file to load
            ~) a list of strings, containing the names of the files that
            should be considered part of the same dataset"""

        self.subdir = os.path.split(dir_name)[1]
        self.file_name = file_name
        self.date = int(self.subdir[0:6])
        self.joined = True
        if type(file_name) is not(list):
            file_name = [file_name]
            self.joined = False

        # Paths and parameter extraction
        self.path = [os.path.join(dir_name, file) for file in file_name]
        self.joined = True
        self.nfiles = len(file_name)
        self.minipath = (self.subdir + r"/" +
                                file_name[0][0:file_name[0].find('.') - 4])
        self.parameter = file_name[0][file_name[0].rindex('_')
                                                    + 1:len(file_name[0])]
        self.power = int(file_name[0][1:file_name[0].index('_') - 2])

        # Data loading
        dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])

        if len(file_name) > 1:
            self.table = np.empty((1), dtype=dt)
            for file in file_name:
                self.table = np.concatenate((self.table,
                                             np.fromfile(file, dtype=dt)))
            self.table = self.table[1:-1]

        else:
            self.table = np.fromfile(file_name, dtype=dt)

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

        # Frame rate extraction from .inf file
        inf_name = self.file_name[0][:self.file_name[0].find('.')] + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
        self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:-2])

    def fit(self, fit_start=0):
        """Histogram fitting"""

        self.fit_start = fit_start

        # Educated guess to initialize the fit
        self.fit_guess = [self.hist[0], 1 / self.mean]

        # Error estimation from Poisson statistics
        sigma = np.sqrt(self.hist[self.fit_start:-1])
        if 0 in sigma:
            sigma = np.asarray([1 if x == 0 else x for x in sigma])

        # Curve fitting
        self.fit_par, self.fit_var = curve_fit(expo,
                                           self.bin_centres[self.fit_start:-1],
                                           self.hist[self.fit_start:-1],
                                           p0=self.fit_guess,
                                           sigma=sigma)
        self.fitted = True

        # Method definitions to make it more verbose
        self.amplitude = self.fit_par[0]
        self.inv_tau = self.fit_par[1]
        if self.parameter in ['offtimes', 'ontimes']:
            self.inv_tau = self.inv_tau * self.frame_rate

        # This will be useful later
        self.results = (self.date, self.power, self.inv_tau)

    def plot(self):
        """Data plotting.
        If the data was fitted, the fitting function is plotted too."""

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.minipath)
        self.ax.bar(self.bin_centres, self.hist, self.bin_width, alpha=0.5,
                    label='_nolegend_')
        self.ax.set_xlabel(self.parameter)
        self.ax.grid(True)
        self.ax.text(0.75 * self.ax.get_xlim()[1], 0.2 * self.ax.get_ylim()[1],
        "Number of counts:\n" + str(self.table.size),
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white'))

        # If the histogram was fit, then we plot also the fitting exponential
        if self.fitted:
            hist_fit = expo(self.bin_centres, *self.fit_par)
            self.ax.plot(self.bin_centres[self.fit_start:-1],
                     hist_fit[self.fit_start:-1],
                     color='r', lw=3,
                     label="y = A * exp(-inv_tau * x)\nA = {}\ninv_tau = {}\n"
                             "tau = {}\npower = {}"
                     .format(int(self.amplitude), self.inv_tau,
                             1 / self.inv_tau, self.power))
            self.ax.legend()

        plt.show()

    def save(self):
        """Save in disk the results of the fitting of this instance of Data"""

        if self.fitted:

            print("Saving...")
            os.chdir(os.path.split(os.getcwd())[0])
            store_name = self.parameter + "_vs_power"
            try:
                if not(os.path.isfile(store_name)):
                    store_file = hdf.File(store_name, "w")
                    store_file.create_dataset("date",
                                          data=np.array([self.date]),
                                          maxshape=(None,))
                    store_file.create_dataset("power",
                                          data=np.array([self.power]),
                                          maxshape=(None,))
                    store_file.create_dataset("inv_tau",
                                          data=np.array([self.inv_tau]),
                                          maxshape=(None,))

                else:

                    # TODO: Check if already saved, change shape
                    store_file = hdf.File(store_name, "r+")
                    prev_size = store_file["date"].size
                    store_file["date"].resize((prev_size + 1,))
                    store_file["date"][prev_size] = self.date
                    store_file["power"].resize((prev_size + 1,))
                    store_file["power"][prev_size] = self.power
                    store_file["inv_tau"].resize((prev_size + 1,))
                    store_file["inv_tau"][prev_size] = self.inv_tau

                store_file.close()

            except:
                print("Unexpected error:", sys.exc_info()[0])
                store_file.close()

        else:
            print("Can't save results, Data not fitted")


def save_folder(store_name, results):
    """Saves the fitting results of the analysis of all files in a folder.
    results = [dates, powers, inv_tau]"""

    print("Saving...")
    cwd = os.getcwd()
    os.chdir(os.path.split(cwd)[0])
    nfiles = results.size

    # If file doesn't exist, it's created and the datasets are added
    try:
        if not(os.path.isfile(store_name)):
            store_file = hdf.File(store_name, "w")
            store_file.create_dataset("sw_results",
                                      data=results,
                                      maxshape=(None,))
        else:
            store_file = hdf.File(store_name, "r+")
            prev_size = store_file["sw_results"].size

            # Check if this data was already saved
            prev_results = store_file.value
            # List of True and Falses indicating if the row is already present
            # in prev_results
            exists = [any(np.equal(prev_results.tolist(),results_i).all(1)) for results_i in results]

            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
            # TODO: keep going

            if any(np.equal(data,results).all(1)):
                print("These results where already saved, "
                "we're not doing that again.")

            else:
                store_file["date"].resize((prev_size + nfiles,))
                store_file["date"][prev_size:] = results[0]
                store_file["power"].resize((prev_size + nfiles,))
                store_file["power"][prev_size:] = results[1]
                store_file["inv_tau"].resize((prev_size + nfiles,))
                store_file["inv_tau"][prev_size:] = results[2]

        store_file.close()

    except:
        print("Unexpected error:", sys.exc_info()[0])
        store_file.close()

    os.chdir(cwd)

def analyze_folder(parameters, from_bin=0):

    dir_name, files_list = load_dir()

    # Conversion of parameters and from_bin to list and array
    if type(parameters) is not(list):
            parameters = [parameters]

    if from_bin == 0:
        from_bin = np.zeros((len(parameters), ))

    for parameter in parameters:

        file_list = []
        for item in files_list:
            file_list.append([item_i for item_i in item if
                                            item_i.endswith("_" + parameter)])

        nfiles = len(file_list)

        # Creation of an instance of Data() for each file
        data_list = [Data() for file in file_list]

        # Structured array for the results of the folder
        r_dtype = np.dtype([('date', int),
                            ('power', int),
                            ('inv_tau', float)])
        folder_results = np.zeros(nfiles, dtype=r_dtype)

        # Process all files in a loop
        for i in range(nfiles):

            # Procedures
            data_list[i].load(dir_name, file_list[i])
            data_list[i].fit(from_bin[parameters.index(parameter)])
            data_list[i].plot()

            # Saving
            folder_results[i] = data_list[i].results

        # Results printing
        print(parameter + " analyzed in " + dir_name)

        # If the plots are ok, Save results
        save = input("Save results? (y/n) ")
        if save == 'y':
            store_name = parameter + "_vs_power.hdf5"
            print(folder_results)
            save_folder(store_name, folder_results)

def load_results(parameter, inv=True,
                 load_dir='\\\\hell-fs\\STORM\\Switching\\'):
    """Load results held in 'parameter'_vs_power.hdf5 file"""

    store_name = parameter + "_vs_power.hdf5"
    os.chdir(load_dir)

    if os.path.isfile(store_name):

        # Load data from HDF5 file
        infile = hdf.File(store_name, "r")
        dates = infile["date"].value
        powers = infile["power"].value
        inv_taus = infile["inv_tau"].value
        n_dtakes = infile["date"].size
        infile.close()

        # Reshape for better table visualization
        np.set_printoptions(suppress=True)
        dates_col = np.array(dates.reshape(n_dtakes, 1))
        powers_col = np.array(powers).reshape(n_dtakes, 1)
        inv_taus_col = np.array(inv_taus).reshape(n_dtakes, 1)
        data = np.hstack((dates_col, powers_col, inv_taus_col))

        # Plot
        if inv:

            plt.scatter(powers, inv_taus)

        else:

            plt.scatter(powers, 1 / inv_taus)

        plt.show()

        return data

    else:

        print("File " + store_name + " not found in " + os.getcwd())

    ### TODO: put units to work


if __name__ == "__main__":



    parameter = ['ontimes', 'photons']
    first_bin = [3, 3]

    import switching_analysis.analysis as sw
#    import imp
#    imp.reload(sw)


#    dir_name, return_list = sw.load_dir()

    sw.analyze_folder(parameter, first_bin)
    print(os.getcwd())
    sw.load_results(parameter)