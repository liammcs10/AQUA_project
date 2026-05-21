'''

Utils script with useful analysis functions when studying AQUA neuronss


'''

## global imports
import numpy as np
import pandas as pd
import brian2
from tqdm import tqdm

from scipy.signal import find_peaks, peak_prominences


def convert_to_biexponential_peak(net_current, t1, t2):
    '''
    Calculate the peak current through the biexponential autapse model
    based on the net current through the reference exponential decay autapse.
    '''

    return (net_current / (t2 - t1)) * ((t1/t2)**(t1/(t2 - t1)) - (t1/t2)**(t2/(t2 - t1)))

def convert_spikes_to_aqua(spike_train):
    """
    Converts a SpikeMonitor.spike_trains() output to the same output from the AQUA class
    * spikemon has time in seconds, time is converted to ms here.

    """
    spikes = []
    for key in spike_train.keys():
        spikes.append(list(spike_train[key]*1000))
    
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

def get_ISI_time_series(spikes, N_iter, dt):
    '''
    Returns a time series in which each point represents the ISI between
    the closest spikes to that point in time. 
    '''
    # create an ISI plot....
    ISI_trace = np.zeros((len(spikes), N_iter))
    ISIs = np.diff(spikes, axis = 1)

    for n, row in enumerate(spikes):
        for i in range(len(row[~np.isnan(row)])-1):

            idx_1 = int(row[i]/dt)
            idx_2 = int(row[i+1]/dt)
            ISI_trace[n, idx_1:idx_2] = ISIs[n, i]

    return ISI_trace


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


def van_rossum_dist(spikes1, spikes2, filter = None):
    '''
    Calculates the van Rossum distance given 2 binary spike time series and a filter
    
    '''
    if filter is None:  # assume inputs already convolved if no filter is passed
        return np.sum((spikes1 - spikes2)**2)
    else:
        s1_filtered = np.convolve(spikes1, filter)
        s2_filtered = np.convolve(spikes2, filter)

    return np.sum((s1_filtered - s2_filtered)**2)

def rolling_VR_dist(spikes1, spikes2, filter, window = 500):
    '''
    Calculates the rolling Van Rossum distance for a time series of synaptic currents
    
    Params
    - - - 
    spikes1, spikes2:       array
                            binary time series of spikes
    window:                 int
                            number of time steps per window

    '''
    assert len(spikes1) == len(spikes2), "syn1 and syn2 should be the same length"

    T = len(spikes1) # duration of the time series

    time_VR = np.zeros(T)

    # convolve spikes
    s1_filtered = np.convolve(spikes1, filter)
    s2_filtered = np.convolve(spikes2, filter)

    for t in range(T - window):

        #time_VR[t] = van_rossum_dist(spikes1[t+window//2:t+window], spikes2[t+window//2:t+window], filter)
        time_VR[t+window//2] = van_rossum_dist(s1_filtered[t:t+window], s2_filtered[t:t+window])
    
    return time_VR


def analyze_isi_peaks(counts, bin_edges, prominence = None, distance = None):
    """
    Finds peaks in an ISI histogram and calculates the mean and std 
    of the ISI values contributing to each peak region.
    
    Parameters:
    -----------
    counts : array-like
        The heights of the histogram bins.
    bin_edges : array-like
        The edges of the bins (length should be len(counts) + 1).
    prominence : float
        Required prominence of peaks (helps filter background noise).
    width : int
        Required width of peaks in terms of number of bins.
        
    Returns:
    --------
    results : list of dicts
        Each dict contains 'peak_isi', 'mean', and 'std' for a detected peak.
    """
    if prominence is None and distance is None:
        # calculated to match the plot_ISI_w_peaks
        prominence = 0.1*np.max(counts)
        distance = 0.1*(len(bin_edges)-1)


    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 1. Find peaks based on height/prominence/width
    # Prominence is key for ignoring background noise
    peaks, properties = find_peaks(counts, prominence = prominence, distance = distance)
    # 2. Find the boundaries (valleys) of each peak to isolate the distribution
    # width_heights determines where the "width" of the peak is measured
    results = []
    
    # Get left and right bases (valleys) for each peak
    prominences = properties['prominences']
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']
    
    for i in range(len(peaks)):
        idx = peaks[i]          # index of the peak
        left = left_bases[i]    
        right = right_bases[i]
        
        # Isolate the bins belonging to this specific peak
        peak_bins = bin_centers[left:right+1]
        peak_counts = counts[left:right+1]
        
        # Calculate the weighted mean and std for this local distribution
        # Mean = sum(x * w) / sum(w)
        local_mean = np.sum(peak_bins * peak_counts) / np.sum(peak_counts)
        
        # Std = sqrt(sum(counts * (bins - mean)^2) / sum(counts))
        local_var = np.sum(peak_counts * (peak_bins - local_mean)**2) / np.sum(peak_counts)
        local_std = np.sqrt(local_var)
        
        results.append({
            'peak_isi': bin_centers[idx],
            'mean': local_mean,
            'std': local_std,
            'count_sum': np.sum(peak_counts),
            'prominence': prominences[i]
        })
        
    return results


def isi_local_variation(spikes):
    '''
    Calculate the local variation in the ISI distribution given a spike train.
    This metric is robust to non-stationary spike trains.
    
    Params:
        spikes:         nd-array of spike times, each row is a separate spike train
    '''

    isi = np.diff(spikes, axis = 1)     # the difference is taken along each row
    LV = np.zeros(len(isi))             # one value per spike train
    for k, row in enumerate(isi):
        n = len(row[~np.isnan(row)])    # number of spikes for this neuron
        if n == 0:
            LV[k] = np.nan
        else:            
            LV[k] = (3/(n-1)) * np.sum(((row[:n-1] - row[1:n])/(row[:n-1] + row[1:n]))**2)
    
    return LV
