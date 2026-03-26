'''

Utils script with useful analysis functions when studying AQUA neuronss


'''

import sys
sys.path.append('..//')


### local imports
from batchAQUA_general import *
from AQUA_general import AQUA


## global imports
import numpy as np
import pandas as pd



def embed(X, window):
    ''' reorder the time series X into (N - window) rows of length window '''
    T = np.shape(X)[0]
    Y = np.zeros(((T - window, window)))
    for i in range(T - window):
        Y[i] = X[i: i + window]
    
    return Y

def binarise(spikes):
    ''' returns an binary array where there is a 1 where there is a spike '''



def STA(spikes, I_inj, dt, window = 50):
    '''
    Calculates the spike triggered average for a given spike_train and corresponding injected
    current array.

    params
    - - - 
    spikes:         ndarray (N_neurons, N_spikes)
                    AQUA spike output. Array of spike times padded with nan_values
    I_inj:          ndarray (N_neurons, N_iter)
                    Injected current trace.
    dt:             float
                    simulation timestep
    window:         int
                    number of timesteps to average before each spike
    Returns
    - - - 
    STA:            ndarray (N_neurons, window)
                    Spike-Triggered Average Injected current before a spike

    '''


    N_neurons, N_iter = np.shape(I_inj)


    # ensure the stimulus has 0 mean
    I_inj = np.array([I_inj[n] - np.mean(I_inj[n]) for n in range(N_neurons)])

    #restructure spikes as a binary time series
    binary_spikes = binarise_spikes(spikes, dt, N_iter)
    
    STA = np.zeros((N_neurons, window))
    for n in range(N_neurons):
        X = embed(I_inj[n], window)

        STA[n] = X.T * binary_spikes[n]

    return STA








