"""
A batch simulation version of the AQUA class designed to run on Nvidia GPUs 

Ways to improve:

    - define a Cupy custom kernel which removes the need for a loop over timesteps!!!



"""

import numpy as np
import cupy as cp
from tqdm import tqdm
import time

## Creation of the GPU kernel

PREAMBLE = '''
#include <vector>
#include <cmath>
#include <string>
#include <iostream>

std::vector<double> neuron_model(
std::vec<double> x,
double w_delay, 
double I,
bool isFS,
double C,
double k,
double v_r,
double v_t,
double a,
double b,
double e){

    // Check if the state vector is correctly sized
    if (x.size() != 3) {
        std::cerr << "Error: State vector x must have 3 elements [v, u, w]." << std::endl;
        return {0.0, 0.0, 0.0}; 
    }

    const double v = x[0];
    const double u = x[1];
    const double w = x[2];

    // 1. calculate dv/dt
    double dv = (1.0/C) * (k * (v - v_r) * (v - v_t) - u + w_delay + I);

    // 2. calculate du/dt
    double du

    if (isFS){
        double U;
        if (v < -55.0) {
            U = 0.0;
        } else {
            U = 0.025 * std::pow(v + 55.0, 3.0);
        }

        du = a * (U - u)
    } else {
        du = a * (b * (v - v_r) - u);
    }

    // 3. calculate dw/dt
    double dw = -1.0 * e * w;

    //return array
    return {dv, du, dw}
}

'''

OPERATION = '''
// everything inside the for-loop

if (n <= delay_steps){
    w_tau1 = w_prev[n - delay_steps - 1]
} else {
    w_tau1 = X[2, n - delay_steps - 1]
}

// first RK2 param
k1 = neuron_model(x, w_tau1, I, isFS, C, k, v_r, v_t, a, b, e)

if (n < delay_steps) {
    w_tau2 = w_prev[n - delay_steps]
} else if (delay_steps == 0. || n == delay_steps) {
    w_tau2 = self.x[2] + k1[2] * dt
} else {
    w_tau2 = X[2, std::max(n - delay_steps, 0)]
}

// second RK2 param
k2 = neuron_model(x + dt * k1, w_tau2, I, isFS, C, k, v_r, v_t, a, b, e)

x += dt * (k1 + k2)/2
t += dt

// Check for spike and update
if (x[0] >= v_peak) {
    x[0] = c
    x[1] += d
    x[2] += f
    spikes = t
} else {
    spikes = std::numeric_limits<double>::quiet_NaN()
}

// end of loop, values returned are x, t, spikes

'''


update_kernel = cp.ElementwiseKernel(
    name = 'update_kernel',
    in_params = '''int64 n, float64 dt, bool isFS, float64 k, float64 C, float64 v_r, float64 v_t, float64 v_peak, float64 a, 
    float64 b, float64 c, float64 d, float64 e, float64 f, float64 tau, std::vector<double> x, float64 t, int delay_steps, std::vec w_prev''',
    out_params = '''std::vec x_next, std::vec t_next, std::vec spikes''',
    
    operation = OPERATION,

    preamble = PREAMBLE
)


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
    

    def update_batch(self, N_iter, dt, I_inj):
        """
        Update of the AQUA model using the cupy.ElementwiseKernel 'update_batch'
        """

        delay_steps = (self.tau / dt).astype(int)

        # check injected current is a cupy array
        if isinstance(I_inj, cp.ndarray) == False:
            I_inj = cp.asarray(I_inj)

        if len(w_prev) == 0:    # if no w_prev passed
            w_prev = cp.zeros((self.N_models, (cp.max(delay_steps)).astype(int)))

        # Arrays for storing the updates
        X = cp.zeros((self.N_models, 3, N_iter), dtype = np.float64)
        X[:, :, 0] = self.x

        T = cp.zeros(N_iter)


        x_next = cp.empty_like(self.x)
        t_next = cp.empty_like(self.t)
        spikes = cp.zeros((self.N_models, N_iter))


        for n in tqdm(cp.arange(N_iter)):

            # run kernel
            update_kernel(n, dt, self.isFS, self.k, self.C, self.v_r, self.v_t, self.v_peak, self.a, 
                          self.b, self.c, self.d, self.e, self.f, self.tau, self.x, self.t, delay_steps, w_prev,
                          x_next, t_next, spikes)

            # update variables
            self.x, self.t = x_next, t_next

            # append to history
            X[:, :, n] = self.x
            T[n] = self.t
            spikes[:, n] = spikes

        return X, T, spikes
 

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
    