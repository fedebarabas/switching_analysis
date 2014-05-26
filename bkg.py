# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/federico/.spyder2/.temp.py
"""

import numpy as np
import os

dtype = np.dtype('>u2')

# Filename
#dire = r'/home/federico/data/MPI/140424_at488_neu_mea_poc/'
#filename = 'b760mW_0p8_uv0_280Hz_TIRF_006.dax'
#
#filename = dire + filename


def load_dax(filename, dtype=dtype):

    # INF data extraction
    inf_data = np.loadtxt(os.path.splitext(filename)[0] + '.inf', dtype=str)
    x_size = int(inf_data[8][inf_data[8].find('=') + 1:])
    y_size = int(inf_data[9][inf_data[9].find('=') + 1:])
    n_frames = int(inf_data[29][inf_data[29].find('=') + 1:])
    shape = (n_frames, x_size, y_size)

    # Data loading
    data = np.memmap(filename, dtype=dtype, mode='r')
    data = data.reshape(shape)

    return data


def bkg_sustraction(data):

    # Background evaluation and substraction
    thres = np.percentile(data, 99.5)
    mask = data > thres
    masked = np.ma.masked_array(data, mask)
    frame_mean = np.mean(masked, axis=0)
    deviations = masked - frame_mean
    offset = np.array([np.mean(dev) for dev in deviations])
    bkg = np.array([frame_mean + off for off in offset])

    return data - bkg


def save_dax(data, name, dtype=dtype):

    if data.min() < 0:
        data = data + np.ceil(abs(data.min()))

    data.astype(dtype).tofile(name)


def subs_bkg(filename, onframe=10):

    # Loading
    data = load_dax(filename)

    # Processing
    init = data[:onframe]
    data = bkg_sustraction(data[onframe:])

    # Saving
    save_dax(np.concatenate((init, data)), filename)


if __name__ == "__main__":

    import tkFileDialog as filedialog
    from Tkinter import Tk

    root = Tk()
    filenames = filedialog.askopenfilenames(parent=root)

    for name in filenames:
        subs_bkg(name)
