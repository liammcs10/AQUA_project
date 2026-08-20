'''

Utils script with useful analysis functions when studying AQUA neuronss


'''

## global imports
import numpy as np
import pandas as pd
import brian2
from tqdm import tqdm

from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture



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
    '''Convert padded AQUA spike-time outputs to binary spike trains.'''
    
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
    """Pad an uneven list of lists into a rectangular NumPy array."""
    max_length = max(len(sublist) for sublist in lst)
    if pad_end:     # pad the end of the list
        return np.array([sublist + [pad_value] * (max_length - len(sublist)) for sublist in lst])
    else:           # pad the front of the list
        return np.array([[pad_value] * (max_length - len(sublist)) + sublist for sublist in lst])


def embed(X, window):
    '''Reorder a time series into overlapping windows for history-based analysis.'''
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
    spikes1, spikes2:       1d array
                            binary time series of spikes
    filter:                 
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

''' Add a Schreiber similarity measure here (Should be quite easy!)'''

def schreiber_similarity(spikes1, spikes2, sigma = None, dt = None):
    '''
    Calculate the schreiber similarity for 2 binary spike trains

    Params:
        spikes#:        nd array
                        binary time series representing spike times.
        sigma:          std of the gaussian filter
    
    OUTPUT

        schreiber:      float
                        schreiber similarity metric for the provided time series
    
    '''
    if (sigma is None) and (dt is None):    # if arguments are not passed, use the raw binary spike time series.
        smooth_spike1 = spikes1
        smooth_spike2 = spikes2
    else:
        smooth_spike1 = gaussian_filter(spikes1, sigma = sigma/dt)
        smooth_spike2 = gaussian_filter(spikes2, sigma = sigma/dt)

    # compute components of the schreiber measure
    dot = np.dot(smooth_spike1, smooth_spike2)
    norm1 = np.linalg.norm(smooth_spike1)
    norm2 = np.linalg.norm(smooth_spike2)

    # schreiber is the normalized dot-product
    if (norm1 == 0.) or (norm2 == 0.):
        schreiber = 0.
    else:
        schreiber = dot/(norm1*norm2)

    return schreiber


''' Add rolling schreiber...'''

def rolling_schreiber(spikes1, spikes2, sigma = 1, dt = 1, window = 500):
    '''
    Returns the rolling schreiber similarity measure
    
    '''
    assert len(spikes1) == len(spikes2), "syn1 and syn2 should be the same length"

    T = len(spikes1)        # duration of the time series
    time_schreiber = np.zeros(T)

    # smooth spike trains before hand for a smoother output
    smooth_spk1 = gaussian_filter(spikes1, sigma = sigma/dt)
    smooth_spk2 = gaussian_filter(spikes2, sigma = sigma/dt)

    for t in range(T - window):
        time_schreiber[t+window//2] = schreiber_similarity(smooth_spk1[t:t+window], smooth_spk2[t:t+window])
    
    return time_schreiber


def rolling_euclid_distance(x1, x2, window = 500):
    '''
    Calculate the rolling euclidean distance between 2 time series
    
    '''
    assert len(x1) == len(x2), "syn1 and syn2 should be the same length"

    T = len(x1) # duration of the time series

    time_VR = np.zeros(T)

    for t in range(T - window):

        #time_VR[t] = van_rossum_dist(spikes1[t+window//2:t+window], spikes2[t+window//2:t+window], filter)
        time_VR[t+window//2] = np.linalg.norm(x1[t:t+window] - x2[t:t+window])
    
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
        prominence = 0.45*np.max(counts)
        distance = 1 #0.02*(len(bin_edges)-1)

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


def discover_and_analyze_isi_peaks(spikes, counts, bin_edges, max_peaks=5):
    """
    Automatically detects the number of overlapping peaks in an ISI histogram
    using BIC scoring, then extracts their statistical parameters.
    
    Parameters:
    -----------
    counts : array-like
        The heights of the histogram bins.
    bin_edges : array-like
        The edges of the bins.
    max_peaks : int
        The maximum number of peaks you realistically expect to find.
    """
    # 1. Recreate the underlying sample distribution from the histogram counts
    #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #simulated_data = np.repeat(bin_centers, counts.astype(int)).reshape(-1, 1)
    simulated_data =  spikes[~np.isnan(spikes)].reshape(-1, 1)
    

    if len(simulated_data) == 0:
        return []

    # 2. Test different numbers of peaks and track their BIC scores
    bic_scores = []
    models = []
    candidate_peak_counts = range(1, max_peaks + 1)
    
    for k in candidate_peak_counts:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(simulated_data)
        bic_scores.append(gmm.bic(simulated_data))
        models.append(gmm)
    
    # 3. Select the model with the LOWEST BIC score
    best_model_idx = np.argmin(bic_scores)
    best_model = models[best_model_idx]
    optimal_n_peaks = candidate_peak_counts[best_model_idx]
    
    # 4. Extract parameters from the winning model
    means = best_model.means_.flatten()
    std_devs = np.sqrt(best_model.covariances_).flatten()
    weights = best_model.weights_
    
    # Sort chronologically by mean ISI
    sort_idx = np.argsort(means)
    
    results = []
    for idx in sort_idx:
        results.append({
            'peak_isi': means[idx],
            'mean': means[idx],
            'std': std_devs[idx],
            'proportion': weights[idx],
            'estimated_count': weights[idx] * np.sum(counts)
        })
        
    # Metadata about the discovery process
    metadata = {
        'detected_num_peaks': optimal_n_peaks,
        'bic_scores': dict(zip(candidate_peak_counts, bic_scores))
    }
    
    return results, metadata


def BGMM_peak_finding(isis, counts, max_peaks=10, weight_threshold=0.05):
    """
    Automatically detects overlapping peaks in an ISI distribution using a 
    SINGLE Bayesian Gaussian Mixture Model by filtering out low-weight clusters.
    """
    # 1. Recreate/clean the underlying sample distribution
    simulated_data = isis[~np.isnan(isis)].reshape(-1, 1) 

    if len(simulated_data) == 0:
        return [], {}

    # 2. Fit a single BGMM with the maximum upper bound
    # Using a Dirichlet Process weight concentration prior
    bgmm = BayesianGaussianMixture(
        n_components=max_peaks, 
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior = 0.00001,
    )
    bgmm.fit(simulated_data)
    
    # 3. Extract raw parameters
    means = bgmm.means_.flatten()
    # Note: covariances_ shape depends on covariance_type (default is 'full')
    # For 1D data, we squeeze it to get variance, then square root for STD
    std_devs = np.sqrt(np.squeeze(bgmm.covariances_))
    weights = bgmm.weights_
    
    # 4. Filter out 'dead' components that the Bayesian prior eliminated
    active_idx = np.where(weights > weight_threshold)[0]
    
    # Sort the surviving peaks chronologically by mean ISI
    sort_idx = active_idx[np.argsort(means[active_idx])]
    
    results = []
    for idx in sort_idx:
        results.append({
            'peak_isi': means[idx],
            'mean': means[idx],
            'std': std_devs[idx],
            'proportion': weights[idx],
            'estimated_count': weights[idx] * np.sum(counts)
        })
        
    metadata = {
        'detected_num_peaks': len(results),
        'active_weights': weights[sort_idx].tolist()
    }
    
    return results, metadata


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
