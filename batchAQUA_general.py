"""
A batch simulation version of the AQUA class. 


"""

import numpy as np
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
        self.name = np.array([p['name'] for p in params_list])
        self.k = np.array([p['k'] for p in params_list])
        self.C = np.array([p['C'] for p in params_list])
        self.v_r = np.array([p['v_r'] for p in params_list])
        self.v_t = np.array([p['v_t'] for p in params_list])
        self.v_peak = np.array([p['v_peak'] for p in params_list])
        self.a = np.array([p['a'] for p in params_list])
        self.b = np.array([p['b'] for p in params_list])
        self.c = np.array([p['c'] for p in params_list])
        self.d = np.array([p['d'] for p in params_list])
        self.e = np.array([p['e'] for p in params_list])
        self.f = np.array([p['f'] for p in params_list])
        self.tau = np.array([p['tau'] for p in params_list])
        #self.E_syn = np.array([p['E_syn'] for p in params_list])

        self.x = np.zeros((self.N_models, 3))
        self.t = np.zeros(self.N_models)
    
    def Initialise(self, x_start, t_start):
        """
        Apply initial conditions to each neuron.

        IN
            x_start:        numpy array (N_models, 3)
            t_start:        numpy array (N_models)
        
        """
        self.x = np.array(x_start)
        self.t = np.array(t_start)
    
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

        
        du = np.zeros(np.shape(u))
        # FS neurons have a nonlinear u-nullcline.

        cond_FS_hyperpolarized = (np.core.defchararray.find(self.name, "FS")!=-1) & (v < -55)
        cond_FS_depolarized = (np.core.defchararray.find(self.name, "FS")!=-1) & (v >= -55)
        cond_notFS = (np.core.defchararray.find(self.name, "FS")==-1)
        # update FS neuron
        du[cond_FS_hyperpolarized] = self.a[cond_FS_hyperpolarized] * (-1. * u[cond_FS_hyperpolarized]) # where neuron is FS and v < -55
        du[cond_FS_depolarized] = self.a[cond_FS_depolarized] * (0.025 * (v[cond_FS_depolarized] + 55.)**3 - u[cond_FS_depolarized])
        # all other neurons are normal
        du[cond_notFS] = self.a[cond_notFS] * (self.b[cond_notFS] * (v[cond_notFS] - self.v_r[cond_notFS]) - u[cond_notFS])
        du[cond_notFS] = self.a[cond_notFS] * (self.b[cond_notFS] * (v[cond_notFS] - self.v_r[cond_notFS]) - u[cond_notFS])
        
        dw = -1 * self.e * w

        return np.array([dv, du, dw]).T
    
    def update_batch(self, dt, N_iter, I_inj, w_prev = []):
        """
        Simulates the response of all neurons to I_inj

        IN
            dt:         time step in ms
            N_iter:     total number of time_steps
            I_inj:      timeseries of injected currents, shape: (N_models, N_iter)
            w_prev:     array, autapse currents prior to sim start - shape: (N_models, delay_steps)

        OUT:
            X:          value of all membrane variables through the trial
                        shape: (N_models, N_iter)
            T:          corresponding time values, shape: (N_iter)
            spikes:     spike times for each neuron, shape: (N_models, )
        
        """

        delay_steps = (self.tau / dt).astype(int)

        if len(w_prev) == 0:
            w_prev = np.zeros(shape = (self.N_models, np.max(delay_steps))) # assume no prior spikes


        X = np.zeros((self.N_models, 3, N_iter), dtype = np.float64)
        X[:, :, 0] = self.x    # (N_models, 3, 1)

        T = np.linspace(0, (N_iter - 1) * dt, N_iter)
        spike_times = [[] for _ in range(self.N_models)]


        for n in tqdm(range(1, N_iter)):  # each neuron updated simultaneously with vectorization

            if n <= np.max(delay_steps): # early in sim
                w_tau1 = np.zeros(self.N_models)                
                tau_idx = np.nonzero(~(n <= delay_steps))                        # indices that need updating
                prev_idx = np.nonzero(n <= delay_steps)                          # where delay_steps extends prior to the sim
                w_tau1[tau_idx] = X[tau_idx, 2, n - delay_steps[tau_idx] - 1]    # get w at the delay
                w_tau1[prev_idx] = w_prev[prev_idx, n - delay_steps[prev_idx]]
                k1 = self.neuron_model(self.x, w_tau1, I_inj[:, n-1])            # first RK param
                
                w_tau2 = np.zeros(self.N_models)  
                bool_idx0 = np.nonzero(n < delay_steps)
                bool_idx1 = np.nonzero(delay_steps == 0.0)           
                bool_idx2 = np.nonzero(n >= delay_steps)

                w_tau2[bool_idx0] = w_prev[bool_idx0, n - delay_steps[bool_idx0] + 1]
                w_tau2[bool_idx1] = self.x[bool_idx1, 2] + k1[bool_idx1, 2] * dt            # update under first condition
                w_tau2[bool_idx2] = X[bool_idx2, 2, n - delay_steps[bool_idx2]]             # update under other condition
                k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[:, n])               # second RK param
            else: # all neurons are beyond delay steps
                rows = np.arange(np.shape(X)[0])
                w_tau1 = X[rows, 2, n - delay_steps - 1]                        # get w at the delay - should be shape (2, )
                k1 = self.neuron_model(self.x, w_tau1, I_inj[:, n-1])           # first RK param
                
                w_tau2 = np.zeros(self.N_models)  
                bool_idx1 = np.nonzero(delay_steps == 0.0)[0]            
                bool_idx2 = np.nonzero(delay_steps != 0.0)[0] # WRONG
                w_tau2[bool_idx1] = self.x[bool_idx1, 2] + k1[bool_idx1, 2] * dt        # update under first condition
                w_tau2[bool_idx2] = X[bool_idx2, 2, n - delay_steps[bool_idx2]]         # update under other condition
                k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[:, n])           # second RK param

            # update with RK2
            self.x = self.x + dt * (k1 + k2)/2
            self.t = self.t + dt

            #Check for spikes and reset
            idx = np.nonzero(self.x[:, 0] >= self.v_peak) # 1 at indices that need updating
            self.x[idx, 0] = self.c[idx]
            self.x[idx, 1] += self.d[idx]
            self.x[idx, 2] += self.f[idx]

            for i in idx[0]: # loop through the indices that have been updated
                spike_times[i].append(self.t[i]) # append the time of spike.
            
            X[:, :, n] = self.x

        spike_times = pad_list(spike_times)     # create a numpy array of fixed dimension
    
        return X, T, spike_times

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

def pad_list(lst, pad_value=np.nan):
    max_length = max(len(sublist) for sublist in lst)
    return np.array([sublist + [pad_value] * (max_length - len(sublist)) for sublist in lst])