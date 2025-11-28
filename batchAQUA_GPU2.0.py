"""
A batch simulation version of the AQUA class designed to run on Nvidia GPUs 

Ways to improve:

    - define a Cupy custom kernel which removes the need for a loop over timesteps!!!



"""

import numpy as np
import cupy as cp
from tqdm import tqdm
import time

class batchAQUA_GPU:

    def __init__(self, params_list):
        """
        A copy of the AQUA_general class optimized for batch simulations. 
        
        IN
            params_list:    list of dictionaries 
                            each dict represents 1 set of neuron params
                            keys correspond to parameter names
        
        Creates a set of N neuron models with the params needed.

        """
        self.N_models = len(params_list)
        # store the indices where an 'FS' neuron is being simulated.
        self.name = np.array([p['name'] for p in params_list])
        whereFS = (np.char.find(self.name, "FS")!=-1)     # boolean array storing where an FS neuron is simulated
        self.isFS = cp.asarray(whereFS)
        self.k = cp.array([p['k'] for p in params_list])
        self.C = cp.array([p['C'] for p in params_list])
        self.v_r = cp.array([p['v_r'] for p in params_list])
        self.v_t = cp.array([p['v_t'] for p in params_list])
        self.v_peak = cp.array([p['v_peak'] for p in params_list])
        self.a = cp.array([p['a'] for p in params_list])
        self.b = cp.array([p['b'] for p in params_list])
        self.c = cp.array([p['c'] for p in params_list])
        self.d = cp.array([p['d'] for p in params_list])
        self.e = cp.array([p['e'] for p in params_list])
        self.f = cp.array([p['f'] for p in params_list])
        self.tau = cp.array([p['tau'] for p in params_list])
        #self.E_syn = np.array([p['E_syn'] for p in params_list])

        self.x = cp.zeros((self.N_models, 3))
        self.t = cp.zeros(self.N_models)
        self.T_FULL = False     # whether to generate a full (N_neurons, N_iter) ndarray of times
    
    def Initialise(self, x_start, t_start):
        """
        Apply initial conditions to each neuron.

        IN
            x_start:        numpy array (N_models, 3)
            t_start:        numpy array (N_models)
        
        """
        self.x = cp.array(x_start, dtype = np.float64)
        self.t = cp.array(t_start, dtype = np.float64)

        if np.unique(t_start) != 1: self.T_FULL = True
    
    def neuron_model(self, x, w_delay, I):
        """
        update rule for the izhikevich neuron with autapse

        IN
            x:          value of membrane variables, shape: (N_models, 3)
            w_delay:    autaptic currents at the time delay, shape: (N_models)
            I:          injected current, shape: (N_models)

        OUT
            arr:        numpy array representing all update values.
                        same shape as x.
        """
        v, u, w = x.T

        dv = (1/self.C) * (self.k * (v - self.v_r) * (v - self.v_t) - u + w_delay + I)
        
        # du = cp.zeros(cp.shape(u))
        # FS neurons have a nonlinear u-nullcline.

        cond_FS_hyperpolarized = self.isFS & (v < -55)  # is FS and hyperpolarized
        cond_FS_depolarized = self.isFS & (v >= -55)    # is FS and depolarized 
        cond_notFS = ~self.isFS # not FS
        # update FS neuron
        #du[cond_FS_hyperpolarized] = self.a[cond_FS_hyperpolarized] * (-1. * u[cond_FS_hyperpolarized]) # where neuron is FS and v < -55, U = 0
        #du[cond_FS_depolarized] = self.a[cond_FS_depolarized] * (0.025 * (v[cond_FS_depolarized] + 55.)**3 - u[cond_FS_depolarized])
        du_FS_hyp = cp.multiply(self.a, cond_FS_hyperpolarized) * (-1. * cp.multiply(u, cond_FS_hyperpolarized))  # where FS with v < -55, U = 0.
        du_FS_dep = cp.multiply(self.a, cond_FS_depolarized) * (0.025 * (cp.multiply(v, cond_FS_depolarized)) + 55.)**3 - cp.multiply(u, cond_FS_depolarized)
        # all other neurons are normal
        #du[cond_notFS] = self.a[cond_notFS] * (self.b[cond_notFS] * (v[cond_notFS] - self.v_r[cond_notFS]) - u[cond_notFS])
        du_not_FS = cp.multiply(self.a, cond_notFS) * (cp.multiply(self.b, cond_notFS) * (cp.multiply(v, cond_notFS) - cp.multiply(self.v_r, cond_notFS)) - cp.multiply(u, cond_notFS))

        du = du_FS_hyp + du_FS_dep + du_not_FS      # merge du

        dw = -1 * self.e * w

        return cp.array([dv, du, dw]).T
    
    ## Creation of the GPU kernel

    update_batch = cp.ElementwiseKernel(
        in_params = '''bool isFS, float64 k, float64 C, float64 v_r, float64 v_t, float64 v_peak, float64 a, 
        float64 b, float64 c, float64 d, float64 e, float64 f, float64 tau, std::vector x, float64 t, int delay_steps'''


    )






 

    def get_params(self, i):
        # returns the dictionary of params for neuron (row) i

        dict = {'name': self.name[i],
                'k': self.k[i],
                'C': self.C[i],
                'v_r': self.v_r[i],
                'v_t': self.v_t[i],
                'v_peak': self.v_peak[i],
                'a': self.a[i],
                'b': self.b[i],
                'c': self.c[i],
                'd': self.d[i],
                'e': self.e[i],
                'f': self.f[i],
                'tau': self.tau[i]}

        return dict
    
def get_spike_times(X, T, v_peak):
    # extract times where spikes occur.
    N_models, _, N_iter = np.shape(X)

    print(f"N_models: {N_models}")
    print(f"N_iter: {N_iter}")

    spike_times = [[] for _ in range(N_models)]

    spike_mask = (X[:, 0, :] <= v_peak[:, np.newaxis])
    print(np.shape(spike_mask))

    model_idx, time_idx = np.nonzero(spike_mask)
    print(model_idx)

    spike_times_flat = T[time_idx]       

    for model_i, time_val in zip(model_idx, spike_times_flat):
        spike_times[model_i].append(time_val)

    spike_times = pad_list(spike_times)
    return spike_times

def pad_list(lst, pad_value=np.nan, pad_end = True):
    max_length = max(len(sublist) for sublist in lst)
    if pad_end:     # pad the end of the list
        return np.array([sublist + [pad_value] * (max_length - len(sublist)) for sublist in lst])
    else:           # pad the front of the list
        return np.array([[pad_value] * (max_length - len(sublist)) + sublist for sublist in lst])
    