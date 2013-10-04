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

    file_list = glob("*.sw_" + parameter)

    return dir_name, file_list

class Data:
    """Methods for analyzing the switching dynamics data"""

    def load(self, file_name, bins=50):
        """Data loading"""

        # Paths and parameter extraction
        self.file_name = file_name
        tmp_path = os.path.split(file_name)
        tmp_path2 = os.path.split(tmp_path[0])
        self.path = tmp_path2[1] + r"/" + tmp_path[1]
        self.parameter = file_name[file_name.rindex('_') + 1:len(file_name)]
        self.power = tmp_path[1][1:tmp_path[1].index('_') - 2]
        self.date = tmp_path2[1][0:6]

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
        self.ax.set_title(self.path)
        self.ax.bar(self.bin_centres, self.hist, self.bin_width, alpha=0.5,
                    label='_nolegend_')
        self.ax.set_xlabel(self.parameter)
        self.ax.grid(True)

        # If the histogram was fit, then we plot also the fitting exponential
        if self.fitted:
            hist_fit = expo(self.bin_centres, *self.fit_par)
            self.ax.plot(self.bin_centres[self.fit_start:-1],
                     hist_fit[self.fit_start:-1],
                     color='r', lw=3,
                     label="y = A * exp(-inv_tau * x)\nA = {}\n1/inv_tau = {}"
                     .format(int(self.amplitude), int(self.inv_tau)))
            self.ax.legend()


if __name__ == "__main__":

    dir_name, file_list = load_dir()

    # Creation of an instance of Data() for each file
    data_list = [Data() for file in file_list]

    powers = []
    results = []
    dates = []

    # Process all files in a loop
    for i in range(len(file_list)):
        data_list[i].load(os.path.join(dir_name, file_list[i]))
        data_list[i].fit(2)
        powers.append(data_list[i].power)
        results.append(data_list[i].tau)
        dates.append(data_list[i].date)

    print(data_list[0].parameter + " analyzed in " + dir_name)
    print(results)