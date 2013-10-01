# -*- coding: utf-8 -*-
#! python3
"""
Created on Wed Sep 25 17:23:02 2013

@author: fbaraba
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Data:
    """Methods for analyzing the switching dynamics data"""

#    def __init__(self):

    def load(self, file_name):
        """Data loading"""

        self.file_name = file_name
        path_list = file_name.split(os.sep)
        parameter = file_name[file_name.rindex('_') + 1:len(file_name)]
        dt = np.dtype([(parameter, '<f4'), ('molecules', '<f4')])
        table = np.fromfile(file_name, dtype=dt)

        return table

    def plot(self, table):

        # HISTOGRAM
        hist, bin_edges = np.histogram(table[parameter], bins=50)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]

        # PLOT THE HISTOGRAM
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        plt.title(path_list[-2] + r"/" + path_list[-1])
        plt.bar(bin_centres, hist, width, alpha=0.5, label=parameter)
        plt.xlabel(parameter)
        plt.grid(True)
        plt.legend()

        #plt.show()
        return fig

    def fit(self, table, ):
        ## HISTOGRAM FITTING
        #def expo(x, a, b):
            #return a * np.exp(-x * b)

        #guess = [hist[0], 1 / table[parameter].mean()]
        ## Should I use the sqrt as a sigma for the photons?
        #if parameter in ['photons', 'totalphotons']:
            #sigma = np.sqrt(hist[5:30])
            #if 0 in sigma:
                #sigma = np.asarray([1 if x == 0 else x for x in sigma])
            #coeff, var_matrix = curve_fit(expo, bin_centres[5:30], hist[5:30],
                #p0=guess, sigma=sigma)
        #else:
            #coeff, var_matrix = curve_fit(expo, bin_centres[5:30], hist[5:30],
                #p0=guess)
        ##coeff, var_matrix = curve_fit(expo, bin_centres[5:30], hist[5:30], p0=guess)
        #hist_fit = expo(bin_centres, *coeff)

        #print(guess)
        #print('Fitted amplitude = ', coeff[0])
        #print('Fitted lambda = ', coeff[1])
        #print('Fitted 1/lambda = ', 1 / coeff[1])

        ## PLOT
        #width = bin_edges[1] - bin_edges[0]
        #plt.title(path_list[-2] + r"/" + path_list[-1])
        #plt.bar(bin_centres, hist, width, alpha=0.5, label='_nolegend_')
        #plt.plot(bin_centres[5:30], hist_fit[5:30], color='r', lw=3,
            #label="y = A * exp(- B * x)\nA = {}\n1/B = {}"
            #.format(int(coeff[0]), int(1 / coeff[1])))
        #plt.xlabel(parameter)
        #plt.grid(True)
        #plt.legend()

        #plt.show()

if __name__ == "__main__":
    datos = Data()
    stringi = (r"\\hell-fs\STORM\Switching\130612_sdab_a647_poc_mea"
    r"\r60mw_uv0p1_113hz_000.sw_photons")
    fig = datos.load(stringi)
    fig.show()
    sys.exit(0)




