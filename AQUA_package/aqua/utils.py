'''

Utils script with useful analysis functions when studying AQUA neuronss


'''

## global imports
import numpy as np
import pandas as pd
from tqdm import tqdm




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

    spike_train = np.zeros((N_neurons, N_iter))

    for n in range(N_neurons):
        spike_idx = (spikes[n][~np.isnan(spikes[n])] / dt).astype(int)       # converts spike times to timesteps
        
        spike_train[n][spike_idx] = 1.

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
    for i in range(0, T-window):
        Y[i] = X[i: i + window]     # get the 'window' time bins before 'i'
    
    return Y



def STA(spikes, I_inj, dt, window=50):
    '''
    Calculates the spike triggered average for a given spike_train and corresponding injected
    current array.

    params
    - - - 
    spikes:     ndarray (N_neurons, N_spikes)
                AQUA spike output. Array of spike times padded with nan_values
    I_inj:      ndarray (N_neurons, N_iter)
                Injected current trace.
    dt:         float
                simulation timestep
    window:     int
                number of timesteps to average before each spike
    
    Returns
    - - - 
    STA:        ndarray (N_neurons, window)
                Spike-Triggered Average Injected current before a spike
    '''
    print('- - STA - - ')
    N_neurons, N_iter = np.shape(I_inj)

    # 1. Vectorized zero-mean across the iter axis (eliminates the list comprehension)
    I_inj = I_inj - np.mean(I_inj, axis=1, keepdims=True)

    STA = np.zeros((N_neurons, window))
    
    # Pre-calculate the relative window offsets [-window, ..., -2, -1]
    offsets = np.arange(-window, 0)
    
    for n in tqdm(range(N_neurons)):
        # 2. Extract valid spike times and convert to timestep indices
        spike_times = spikes[n]
        spike_idx = (spike_times[~np.isnan(spike_times)] / dt).astype(int)

        # 3. Filter spikes to ensure we have a full window before them
        valid_idx = spike_idx[(spike_idx >= window) & (spike_idx < N_iter)]

        if len(valid_idx) == 0:
            continue

        # 4. Use advanced indexing to fetch only the relevant windows
        # Broadcasting creates a 2D array of indices of shape (num_valid_spikes, window)
        window_indices = valid_idx[:, None] + offsets
        
        # 5. Extract the windows and sum across the spike dimension (axis 0)
        STA[n] = np.sum(I_inj[n][window_indices], axis=0)
        
        # NOTE: Your original dot product calculates a SUM, not an average. 
        # If you want the true mathematical average, uncomment the line below:
        STA[n] = STA[n] / len(valid_idx)

    return STA






