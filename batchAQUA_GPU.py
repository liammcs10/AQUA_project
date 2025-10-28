"""
A batch simulation version of the AQUA class. 


"""

import numpy as np
import cupy as cp
from tqdm import tqdm

class batchAQUA:

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
        names = np.array([p['name'] for p in params_list])
        self.name = cp.asarray(names)
        whereFS = (np.char.find(names, "FS")!=-1)     # boolean array storing where an FS neuron is simulated
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
    
    def Initialise(self, x_start, t_start):
        """
        Apply initial conditions to each neuron.

        IN
            x_start:        numpy array (N_models, 3)
            t_start:        numpy array (N_models)
        
        """
        self.x = cp.array(x_start, dtype = np.float64)
        self.t = cp.array(t_start, dtype = np.float64)
    
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

        
        du = cp.zeros(cp.shape(u))
        # FS neurons have a nonlinear u-nullcline.

        cond_FS_hyperpolarized = self.isFS & (v < -55)  # is FS and hyperpolarized
        cond_FS_depolarized = self.isFS & (v >= -55)    # is FS and depolarized 
        cond_notFS = ~self.isFS # not FS
        # update FS neuron
        du[cond_FS_hyperpolarized] = self.a[cond_FS_hyperpolarized] * (-1. * u[cond_FS_hyperpolarized]) # where neuron is FS and v < -55, U = 0
        du[cond_FS_depolarized] = self.a[cond_FS_depolarized] * (0.025 * (v[cond_FS_depolarized] + 55.)**3 - u[cond_FS_depolarized])
        # all other neurons are normal
        du[cond_notFS] = self.a[cond_notFS] * (self.b[cond_notFS] * (v[cond_notFS] - self.v_r[cond_notFS]) - u[cond_notFS])
        du[cond_notFS] = self.a[cond_notFS] * (self.b[cond_notFS] * (v[cond_notFS] - self.v_r[cond_notFS]) - u[cond_notFS])
        
        dw = -1 * self.e * w

        return cp.array([dv, du, dw]).T
    
    def update_batch(self, dt, N_iter, I_inj, w_prev = []):
        """
        Simulates the response of all neurons to I_inj

        IN
            dt:         time step in ms
            N_iter:     total number of time_steps
            I_inj:      timeseries of injected currents, shape: (N_models, N_iter)
            w_prev:     array, autapse currents prior to sim start - shape: (N_models, delay_steps)
                        if different tau, pad this list with np.nan with pad_end = False

        OUT:
            X:          value of all membrane variables through the trial
                        shape: (N_models, N_iter)
            T:          corresponding time values, shape: (N_iter)
            spikes:     spike times for each neuron, shape: (N_models, )
        
        """

        """ CONVERT ALL np.ndarrays to cupy"""

        delay_steps = (self.tau / dt).astype(int)

        # if injected current isn't a cupy array, convert to cupy
        if isinstance(I_inj, cp.ndarray) == False:
            I_inj = cp.asarray(I_inj)
        

        if len(w_prev) == 0:
            w_prev = cp.zeros(shape = (self.N_models, cp.max(delay_steps))) # assume no prior spikes


        X = cp.zeros((self.N_models, 3, N_iter), dtype = np.float64)
        X[:, :, 0] = self.x    # (N_models, 3, 1)

        T = cp.linspace(0, (N_iter - 1) * dt, N_iter)
        spike_times = [[] for _ in range(self.N_models)]


        for n in tqdm(range(1, N_iter)):  # each neuron updated simultaneously with vectorization

            if n <= cp.max(delay_steps): # early in sim

                w_tau1 = cp.zeros(self.N_models)                
                tau_idx = cp.nonzero(~(n <= delay_steps))                        # indices that need updating
                prev_idx = cp.nonzero(n <= delay_steps)                          # where delay_steps extends prior to the sim
                w_tau1[tau_idx] = X[tau_idx, 2, n - delay_steps[tau_idx]-1]       # get w at the delay
                w_tau1[prev_idx] = w_prev[prev_idx, n - delay_steps[prev_idx] - 1]

                
                k1 = self.neuron_model(self.x, w_tau1, I_inj[:, n-1])            # first RK param
                
                w_tau2 = cp.zeros(self.N_models)  
                bool_idx0 = cp.nonzero(n <= delay_steps - 1)         # delay_steps extends before the sim start
                bool_idx1 = cp.nonzero(delay_steps == 0.0)           # case where there is no delay, need to estimate w_tau2
                bool_idx2 = cp.nonzero(n > delay_steps - 1)          # case where w_tau2 is has been calculated previously

                w_tau2[bool_idx0] = w_prev[bool_idx0, n - delay_steps[bool_idx0]]
                w_tau2[bool_idx1] = self.x[bool_idx1, 2] + k1[bool_idx1, 2] * dt            # update under first condition
                w_tau2[bool_idx2] = X[bool_idx2, 2, n - delay_steps[bool_idx2]]             # update under other condition

                k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[:, n])               # second RK param

            else: # all neurons are beyond delay steps
                rows = cp.arange(np.shape(X)[0])            # all rows
                w_tau1 = X[rows, 2, n - delay_steps - 1]                    # get w at the delay - should be shape (2, )
                k1 = self.neuron_model(self.x, w_tau1, I_inj[:, n-1])           # first RK param
                
                w_tau2 = cp.zeros(self.N_models)  
                bool_idx1 = cp.nonzero(delay_steps == 0.0)[0]            
                bool_idx2 = cp.nonzero(delay_steps != 0.0)[0]
                w_tau2[bool_idx1] = self.x[bool_idx1, 2] + k1[bool_idx1, 2] * dt        # update under first condition
                w_tau2[bool_idx2] = X[bool_idx2, 2, n - delay_steps[bool_idx2]]         # update under other condition
                k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[:, n])           # second RK param

            # update with RK2
            self.x = self.x + dt * (k1 + k2)/2
            self.t = self.t + dt

            #Check for spikes and reset
            idx = cp.nonzero(self.x[:, 0] >= self.v_peak) # 1 at indices that need updating
            self.x[idx, 0] = self.c[idx]
            self.x[idx, 1] += self.d[idx]
            self.x[idx, 2] += self.f[idx]
            
            for i in idx[0]: # loop through the indices that have been updated
                spike_times[i].append(self.t[i]) # append the time of spike.
            
            X[:, :, n] = self.x

        spike_times = pad_list(spike_times)     # create a numpy array of fixed dimension

        # convert all outputs back to numpy/cpu
        X_cpu = X.get()
        T_cpu = T.get()
        spike_times_cpu = spike_times.get()
    
        return X_cpu, T_cpu, spike_times_cpu

    def get_params(self, i):
        # returns the dictionary of params for neuron i

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

def pad_list(lst, pad_value=np.nan, pad_end = True):
    max_length = max(len(sublist) for sublist in lst)
    if pad_end:     # pad the end of the list
        return np.array([sublist + [pad_value] * (max_length - len(sublist)) for sublist in lst])
    else:           # pad the front of the list
        return np.array([[pad_value] * (max_length - len(sublist)) + sublist for sublist in lst])
    