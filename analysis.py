# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: Federico Barabas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expo(x, A, inv_tau):
    return A * np.exp(- inv_tau * x)

def load_dir(initialdir='\\\\hell-fs\\STORM\\Switching\\', parameter='photons'):

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
    file_list = glob("*.sw_" + parameter)

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
        self.date = self.subdir[0:6]

        if type(file_name) is list:
            # Paths and parameter extraction
            self.path = [os.path.join(dir_name, file) for file in file_name]
            self.joined = True
            self.nfiles = len(file_name)
            self.minipath = (self.subdir + r"/" +
                                    file_name[0][0:file_name[0].find('.') - 4])
            self.parameter = file_name[0][file_name[0].rindex('_')
                                                        + 1:len(file_name[0])]
            self.power = file_name[0][1:file_name[0].index('_') - 2]

            # Data loading
            dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])
            self.table = np.empty((1), dtype=dt)
            for file in file_name:
                self.table = np.concatenate((self.table,
                                             np.fromfile(file, dtype=dt)))
            self.table = self.table[1:-1]

        else:
            # Paths and parameter extraction
            self.joined = False
            self.minipath = self.subdir + r"/" + file_name
            self.parameter = file_name[file_name.rindex('_') + 1:len(file_name)]
            self.power = file_name[1:file_name.index('_') - 2]

            # Data loading
            dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])
            self.table = np.fromfile(file_name, dtype=dt)

        # Histogram construction
        self.mean = np.mean(self.table[self.parameter])
        self.hist, bin_edges = np.histogram(self.table[self.parameter],
                                            bins=bins,
                                            range=(0, bins * self.mean / 10))
        self.bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_width = bin_edges[1] - bin_edges[0]
        self.fitted = False

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
        self.tau = 1 / self.fit_par[1]

    def plot(self):
        """Data plotting"""

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
                     label="y = A * exp(-inv_tau * x)\nA = {}\ntau = {}"
                     .format(int(self.amplitude), int(self.tau)))
            self.ax.legend()


if __name__ == "__main__":

    dir_name, file_list = load_dir()

    # Creation of an instance of Data() for each file
    data_list = [Data() for file in file_list]

    results = np.empty([len(file_list), 3])

#   Process all files in a loop
    for i in range(len(file_list)):
        data_list[i].load(dir_name, file_list[i])
        data_list[i].fit(2)
        data_list[i].plot()
        results[i] = [data_list[i].date, data_list[i].tau, data_list[i].power]

    print(data_list[0].parameter + " analyzed in " + dir_name)
    print(results)