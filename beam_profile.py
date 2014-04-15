# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014

@author: fbaraba
"""

# Python2-3 compatibility, not tested
from __future__ import division, with_statement, print_function

import numpy as np

# Python2-3 compatiblilty handling
try:
    from tkinter import Tk, filedialog

except ImportError:
    import tkFileDialog as filedialog
    from Tkinter import Tk


initialdir = '\\\\hell-fs\\STORM\\Switching\\data\\'
shapes = (100, 128, 128)
t_on = 10
r_min = (15, 25)
l = 85


def load_files(ask, initialdir=initialdir):

    # Get filenames from user
    try:

        root = Tk()
        file_names = filedialog.askopenfilenames(parent=root,
                                                 initialdir=initialdir,
                                                 title=ask)
        root.destroy()

    except OSError:
        print("No files selected!")

    return file_names.split()


def load_stack(filename, shape=shapes, dtype=np.dtype('>u2')):

    # assuming big-endian
    data = np.memmap(filename, dtype=dtype, mode='r')
    return data.reshape(shape)


def mean_frame(stack, start_frame=0):
    '''Get the mean of all pixels from start_frame'''

    return stack[start_frame:].mean(0)


def get_beam(image):

    hist, edg = np.histogram(image, bins=50)

    # We get the inside of the beam as the pixels with intesity higher than the
    # minimum of the histogram of the image, between the background and the
    # beam distribution
    thres = edg[np.argmin(hist[:np.argmax(hist)])]
    beam_mask = np.zeros(shape=image.shape)
    beam_mask[image < thres] = True

    return np.ma.masked_array(image, beam_mask)


def beam_mean(ask):

    filenames = load_files(ask)

    mean_frames = np.zeros((len(filenames), shapes[1], shapes[2]))

    for i in np.arange(len(filenames)):
        print(filenames[i])
        data = load_stack(filenames[i])
        mean_frames[i] = mean_frame(data, t_on)

    return get_beam(mean_frames.mean(0))


def frame(image, r_min=r_min, l=l):

    return image[r_min[0]:r_min[0] + l, r_min[1]:r_min[1] + l]


if __name__ == "__main__":

#    %load_ext autoreload
#    %autoreload 2

    import sys

    repos = 'P:\\Private\\repos'
    sys.path.append(repos)

    import switching_analysis.beam_profile as bp

    epi_mean = bp.beam_mean('epi files')
    tirf_mean = bp.beam_mean('tirf files')

    tirf_factor = frame(tirf_mean).mean() / frame(epi_mean).mean()
    frame_factor = frame(tirf_mean).mean() / tirf_mean.mean()

    print('tirf factor', tirf_factor)
    print('frame factor', frame_factor)
    print('variance in tirf frame',
          100 * frame(tirf_mean).std() / frame(tirf_mean).mean())

#   plt.imshow(tirf_mean, interpolation='none')
#   plt.colorbar()
