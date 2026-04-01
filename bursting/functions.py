


import numpy as np





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


