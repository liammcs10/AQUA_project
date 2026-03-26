'''

Utils script with useful analysis functions when studying AQUA neuronss


'''

### local imports
from .batchAQUA_general import *
from .AQUA_general import AQUA


## global imports
import numpy as np
import pandas as pd




def convert_spikes_to_aqua(spike_train):
    """
    Converts a SpikeMonitor.spike_trains() output to the same output from the AQUA class
    """
    spikes = []
    for key in spike_train.keys():
        spikes.append(list(spike_train[key]/ms))
    
    spikes = pad_list(spikes)

    return spikes

def binarise_spikes(spikes, dt, N_iter):
    ''' Convert AQUA spike outputs to binary spike trains '''
    N_neurons = np.shape(spikes)[0]

    spike_idx = (spikes / dt).astype(int)     # converts spike times to timesteps

    spike_train = np.zeros((N_neurons, N_iter))

    for n in range(N_neurons):
        spike_train[n][spike_idx[n]] = 1.

    return spike_train

def pad_list(lst, pad_value=np.nan, pad_end = True):
    max_length = max(len(sublist) for sublist in lst)
    if pad_end:     # pad the end of the list
        return np.array([sublist + [pad_value] * (max_length - len(sublist)) for sublist in lst])
    else:           # pad the front of the list
        return np.array([[pad_value] * (max_length - len(sublist)) + sublist for sublist in lst])

def embed(X, window):
    ''' reorder the time series X into (N - window) rows of length window '''
    T = np.shape(X)[0]
    Y = np.zeros(((T - window, window)))
    for i in range(T - window):
        Y[i] = X[i: i + window]
    
    return Y


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
                    This may be a biased estimate of the receptive field depending on the
                    injected current used.

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








