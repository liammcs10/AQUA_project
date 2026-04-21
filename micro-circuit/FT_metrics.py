''' import aqua '''
from aqua.batchAQUA_general import batchAQUA
from aqua.AQUA_general import AQUA
from aqua.utils import * 

'''general imports''' 
import numpy as np
import pandas as pd
from brian2 import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import convolve, windows




def calculate_FT(binary_spikes, dt, filter = None):
    '''
    calculate the FFT of a binary time series representing spiking activity.
    time series is convolved with a filter to smooth it.
    Params:
    - - -
    binary_spikes:          1d array
                            each row is a different binary time series representing spiking
    dt:                     float
                            time step of the simulation
    filter;                 array
                            choice of filter for convolution
    
    Returns
    - - - 
    FFT                     array
                            FFT spectrum
    FFT_frequencies         array
                            frequencies corresponding to each element in FFT
    '''

    if filter is None:
        # if no filter, use a gaussian
        filter = windows.gaussian(M = 10000, std = 100)
        filter /= filter.sum()    # normalise

    # convolve binary spikes
    freq = convolve(binary_spikes, filter)[:np.shape(binary_spikes)[1]]

    #calculate FFT
    FFT = np.fft.fft(freq)

    #calculate corresponding frequencies
    FFT_frequencies = np.fft.fftfreq(len(freq), d = dt/1000)

    return FFT, FFT_frequencies

def calculate_FT_diff(FFT1, FFT2, FFT_freq, freq_cutoff = 100):
    '''
    Calculates difference in the area between both FFT curves. Serves as a metric of the degree of synchrony and entrainment.

    Each FFT curve is normalised before differencing. The maximum possible difference is thus 2, and the minimum is 0.

    Params:
    - - - 
    FFT1, FFT2              array
                            returned from np.fft.fft function
    FFT_freq                array
                            corresponding frequencies
    freq_cutoff             float
                            the cutoff frequency
    
    Returns:
    - - -
    diff                    float
                            difference in areas of the FFT curves up to freq_cutoff
    '''

    cutoff_idx = np.argwhere(FFT_freq == freq_cutoff)[0][0]
    # only take values up to cutoff
    sub_FFT1 = np.abs(FFT1[:len(FFT_freq)//2])[:cutoff_idx]
    sub_FFT2 = np.abs(FFT2[:len(FFT_freq)//2])[:cutoff_idx]
    sub_freq = FFT_freq[:cutoff_idx]

    # normalise each curve
    sub_FFT1 /= sub_FFT1.sum()
    sub_FFT2 /= sub_FFT2.sum()

    # difference
    plt.plot(sub_freq, sub_FFT2 - sub_FFT1, c = 'black')
    plt.plot(sub_freq, np.abs(sub_FFT2 - sub_FFT1), c = 'blue')
    plt.show()

    diff = np.sum(np.abs(sub_FFT2 - sub_FFT1))

    return diff

def rolling_FT_diff(binary_spikes, dt, filter, window):
    '''
    Calculate the rolling Fourier Transform difference to measure how synchrony
    converges over time.

    
    
    '''


