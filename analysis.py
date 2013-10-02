# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: fbaraba
"""

import os
import numpy as np
import matplotlib
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expo(x, A, inv_tau):
    return A * np.exp(-inv_tau * x)

class Data:
    """Methods for analyzing the switching dynamics data"""

    def load(self, file_name, bins=50):
        """Data loading"""

        self.file_name = file_name
        path_list = file_name.split(os.sep)
        self.path = path_list[-2] + r"/" + path_list[-1]
        self.parameter = file_name[file_name.rindex('_') + 1:len(file_name)]
        dt = np.dtype([(self.parameter, '<f4'), ('molecules', '<f4')])
        self.table = np.fromfile(file_name, dtype=dt)

        # HISTOGRAM CONSTRUCTION
        self.mean = np.mean(self.table[self.parameter])
        self.hist, bin_edges = np.histogram(self.table[self.parameter],
                                            bins=bins,
                                            range=(0, bins * self.mean / 10))
        self.bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_width = bin_edges[1] - bin_edges[0]

    def plot(self):
        """Data plotting"""

        # PLOT THE HISTOGRAM
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        plt.title(self.path)
        plt.bar(self.bin_centres, self.hist, self.bin_width, alpha=0.5,
                                                        label=self.parameter)
        plt.xlabel(self.parameter)
        plt.grid(True)
        plt.legend()
        plt.show(block='True')
        self.plot = fig        # not really working

    def fit(self, fit_start=0, quiet=False):
        """Histogram fitting"""

        # Educated guess to initialize the fit
        self.fit_guess = [self.hist[0], 1 / self.mean]
        ## Should I use the sqrt as a sigma for the photons?
        #if parameter in ['photons', 'totalphotons']:
            #sigma = np.sqrt(hist[5:30])
            #if 0 in sigma:
                #sigma = np.asarray([1 if x == 0 else x for x in sigma])
            #self.fit_par, var_matrix = curve_fit(expo, bin_centres[5:30], hist[5:30],
                #p0=self.fit_guess, sigma=sigma)
        #else:
            #self.fit_par, var_matrix = curve_fit(expo, bin_centres[5:30], hist[5:30],
                #p0=self.fit_guess)
        self.fit_par, self.fit_var = curve_fit(expo,
                                               self.bin_centres[fit_start:-1],
                                               self.hist[fit_start:-1],
                                               p0=self.fit_guess)

        # Method definitions to make it more verbose. They are redundant but
        # they will make the code clearer

        self.amplitude = self.fit_par[0]
        self.inv_tau = self.fit_par[1]
        self.tau = 1 / self.fit_par[1]

#        print(self.fit_guess)
#        print('Fitted amplitude = ', self.amplitude)
#        print('Fitted inv_tau = ', self.inv_tau)
#        print('Fitted tau = ', self.tau)

        # PLOT

        if not(quiet):
            matplotlib.use('Agg')

        fig = plt.figure()
        hist_fit = expo(self.bin_centres, *self.fit_par)
        plt.title(self.path)
        plt.bar(self.bin_centres, self.hist, self.bin_width,
                alpha=0.5, label='_nolegend_')
        plt.plot(self.bin_centres[fit_start:-1], hist_fit[fit_start:-1],
                 color='r', lw=3,
                 label="y = A * exp(-inv_tau * x)\nA = {}\n1/inv_tau = {}"
                 .format(int(self.amplitude), int(self.inv_tau)))
        plt.xlabel(self.parameter)
        plt.grid(True)
        plt.legend()
        self.figure = fig

        if not(quiet):
            plt.show(block='True')

if __name__ == "__main__":

    from tkinter import Tk, filedialog
    from glob import glob

    # Get working folder from user
    try:
        root = Tk()
        dirname = filedialog.askdirectory(parent=root,
                                initialdir='\\\\hell-fs\\STORM\\Switching\\',
                                title='Please select a directory')
        root.destroy()
        os.chdir(dirname)
    except OSError:
        print("No folder selected!")

    print(dirname)
    file_list = glob("*.sw_photons")
    print(file_list)

    # Creation of an instance of Data() for each file
    data_list = [Data() for file in file_list]

    # Process all files in a loop
    for i in range(len(file_list)):
        data_list[i].load(os.path.join(dirname, file_list[i]))
        data_list[i].fit(2, True)
        print(data_list[i].tau)

    data_list[2].figure.canvas.draw()
#    plt.draw()