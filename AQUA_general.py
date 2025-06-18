"""
General Izhikevich neuron model. 

Cortical neurons and parameter values.
RS = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -50, 'd': 100}    # Class 1

IB = {'name': 'IB', 'C': 150, 'k': 1.2, 'v_r': -75, 'v_t': -45, 'v_peak': 50,
     'a': 0.01, 'b': 5, 'c': -56, 'd': 130}

CH = {'name': 'CH', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -40, 'd': 100}

LTS = {'name': 'LTS', 'C': 100, 'k': 1, 'v_r': -56, 'v_t': -42, 'v_peak': 35,
     'a': 0.03, 'b': 8, 'c': -50, 'd': 100}   # Border of integrator-resonator (Bogdanov-Takens)

FS = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
     'a': 0.2, 'b': -2, 'c': -45, 'd': 0}   # subcritical Andronov-Hopf (class 2)
    * requires a nonlinear u-nullcline (horizontal in the hyperpolarized range)

MSN = {'name': 'MSN', 'C': 50, 'k': 1, 'v_r': -80, 'v_t': -25, 'v_peak': 40,
     'a': 0.01, 'b': -20, 'c': -55, 'd': 150} # bistable (SN or subAH)(striatum projection neuron)
"""

import numpy as np
from tqdm import tqdm


class AQUA:

    """ This class creates and evolves a single neuron according to the izhikevich model.
     Currently no autapse is implemented.
     Can also extend this code to allow for u, a, and b to be vectors.
    
    """

    def __init__(self, param_dict): 
        """
        Param_dict:
        {'name': 'FS', 'k': k, 'C': C, 'v_r': v_r, 'v_t': v_t, 
        'v_peak': v_peak, 'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'tau': tau}
        
        """
        
        self.N_dim = 3 # by default, always.
        self.name = param_dict['name'] # name of the neuron model 
        self.k, self.C = param_dict['k'], param_dict['C']
        self.v_r, self.v_t, self.v_peak = param_dict['v_r'], param_dict['v_t'], param_dict['v_peak']
        self.a, self.b, self.c, self.d = param_dict['a'], param_dict['b'], param_dict['c'], param_dict['d']
        self.e, self.f, self.tau = param_dict['e'], param_dict['f'], param_dict['tau']
        self.x = np.zeros(3)
        self.t = 0.0

    def Initialise(self, x_start, t_start):
        """
        Set starting values
        x is 3-element [v, u, w]. If no autapse, w is unchanged.
        t is a float.
        """
        self.x[:] = x_start
        self.t = t_start #+ self.tau


    def neuron_model(self, x, w_delay, I):
        """
        Definition of the izhikevich neuron model. 
        v = membrane potential
        u = membrane recovery variable
        w = autaptic current
        """

        v, u, w = x

        dv = (1/self.C) * (self.k * (v - self.v_r) * (v - self.v_t) - u + w_delay + I)
        
        # Different model for FS neurons
        if "FS" in self.name:
            if v < -55:
                U = 0
            else:
                U = 0.025 * (v + 55)**3
            
            du = self.a * (U - u) # u-nullcline is now nonlinear for FS
        else:
            du = self.a * (self.b * (v - self.v_r) - u)
        
        dw = -1 * self.e * w

        return np.array([dv, du, dw]) 
    

    def update_RK2(self, dt, N_iter, I_inj, w_prev = []):
        """
        Simulates the neuron response to a injected current using RK2.
        
        IN:
            dt: timestep
            N_iter: number of iterations
            I_inj: array representing the injecter current [pA] of length N_iter
            w_prev: array (size: int(self.tau/dt)), autapse current prior to sim start.
        OUT
            X: the membrane variables during the simulation
            T: corresponding time vector
            spike_times: array storing the time of spikes. 
        """


        # dt = timestep
        # N_iter = number of iterations
        # I_inj = injected current: len(I_inj) >= N_iter

        delay_steps = int(self.tau / dt)

        if len(w_prev) == 0:
            w_prev = np.zeros(delay_steps) # if not defined, assume no previous spikes and autapse current is 0.0

        # Arrays to be returned, intially filled with all start values
        X = np.zeros((3, N_iter), dtype = np.float64)
        X[:, 0] = self.x    # (3x1)
    
        T = np.linspace(0, (N_iter - 1) * dt, N_iter)
        spike_times = []

        for n in tqdm(range(1, N_iter)): # can't start at 0 because this one is already calculated
            # Calculate the RK update variables.
            # Here, index n-1 corresponds to the 'current' value off which updates need to be estimated.

            if n <= delay_steps: 
                w_tau1 = w_prev[n - delay_steps] # value of autapse current prior to simulation
            else:
                w_tau1 = X[2, n - delay_steps - 1]
            
            k1 = self.neuron_model(self.x, w_tau1, I_inj[n - 1])

            if n < delay_steps: # change 6/17 - 2:38pm
                w_tau2 = w_prev[n - delay_steps + 1]
            elif delay_steps == 0 or n == delay_steps:
                w_tau2 = self.x[2] + k1[2] * dt 
            else:
                w_tau2 = X[2, max(n - delay_steps, 0)]
            
            k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[n])
            
            # Update x using RK2. Increment t.
            self.x += dt * (k1 + k2) / 2
            self.t = self.t + dt

            # Check for spike and reset variables.
            if self.x[0] >= self.v_peak:
                self.x[0] = self.c    # reset membrane potential
                self.x[1] += self.d    # bump membrane reset variable 
                self.x[2] += self.f    # bump autaptic variable
                spike_times.append(self.t)

            X[:, n] = self.x
        
        return X, T, np.array(spike_times)

    def get_params(self):
        # returns the dictionary of params for neuron i

        dict = {'name': self.name,
                'k': self.k,
                'C': self.C,
                'v_r': self.v_r,
                'v_t': self.v_t,
                'v_peak': self.v_peak,
                'a': self.a,
                'b': self.b,
                'c': self.c,
                'd': self.d,
                'e': self.e,
                'f': self.f,
                'tau': self.tau}

        return dict