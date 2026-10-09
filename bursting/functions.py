
import numpy as np
from aqua.utils import analyze_isi_peaks
from scipy.stats import entropy


""" - - - - HELPER FUNCTIONS - - - - """

def get_F(spikes, instant = False):
    """
    Returns an array of the desired firing frequency
    
    :param spikes: array of spike times
    :param instant: boolean, whether to get instantaneous firing frequency or not

    """
    N_neurons = len(spikes)
    F = np.zeros(N_neurons)

    for n in range(N_neurons):
        if np.isnan(spikes[n]).all() or np.sum(~np.isnan(spikes[n])) <= 3:      # if no spikes or 1 spike
            F[n] = np.nan
        else:
            if instant:     # get instant firing frequency
                F[n] = 1000/(spikes[n][1] - spikes[n][0])           # first and second spikes
            else:           # get steady firing frequency (might be same as initial)
                spike_times = spikes[n][~np.isnan(spikes[n])]
                n_spikes = len(spike_times)
                ceil = np.ceil(n_spikes/2)
                freq = 1000/(np.ediff1d(spike_times[-int(ceil):]))
                F[n] = np.max(freq)     # largest firing frequency in the steady-state

    return F

def get_num_peaks(spikes, bins = 300, range = (0, 150), prominence_fraction = None, distance = None):
    """
    returns the number of isi peaks for each neuron in 'spikes'
    """
    N_neurons = len(spikes)
    # convert spikes to ISIs
    isi = np.diff(spikes, axis = 1)
    # bin spikes for peak finding function
    num_peaks = np.zeros(N_neurons, dtype = np.int32)

    for n, row in enumerate(isi):
        counts, bin_edges = np.histogram(row, bins = bins, range = range)
        if prominence_fraction is not None:
            prominence = prominence_fraction * np.sum(counts)
        else:
            prominence = None
        num_peaks[n] = len(analyze_isi_peaks(counts, bin_edges, prominence, distance))

    return num_peaks

def get_entropy(spikes, bins = 300, range = (0, 150)):
    """ 
    Calculate the entropy of the ISI distribution for each neuron in 'spikes'
    """
        # convert spikes to ISIs
    isi = np.diff(spikes, axis = 1)
    spike_entropy = np.zeros(isi.shape[0])

    for n, row in enumerate(isi):
        counts, _ = np.histogram(row, bins = bins, range = range)
        spike_entropy[n] = entropy(counts, base = 2.0)

    return spike_entropy


def cast_to_float(data_dict):
    """
    Casts the values of a dictionary to float if conversion is possible.
    Otherwise, the original value is retained.
    """
    new_dict = {
        key: float(value) 
        if isinstance(value, (int, str)) and value not in ('', None) and is_float(value)
        else value
        for key, value in data_dict.items()
    }
    return new_dict

def is_float(value):
    """Helper function to safely check if a string can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


