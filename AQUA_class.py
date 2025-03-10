"""

- Might be interesting to extend the model so that parameters can be vectors allowing for multiple timescales. 

Can add an additional class which represents the network of izhikevich neurons. 
In this case, simulations will want to make use of numpy vectorization to produce optimal simulations. 
    - This is very useful when it comes to batch simulations and GPU computing (can use CuPy to make use of numpy vectorization but on GPUs)




"""


import numpy as np


class AQUA:

    """ This class creates and evolves a single neuron according to the izhikevich model.
     Currently no autapse is implemented.
     Can also extend this code to allow for u, a, and b to be vectors.
    
    """

    def __init__(self, a, b, c, d, e = 0., f = 0., tau = 0.):
        """
        model params
        a, b, c, d are izhikevich params for cortical neurons
        e, f, tau, are autaptic parameters.
        e = inverse decay time constant of autaptic variable
        f = reset value of autaptic variable
        tau = time delay of autapse
        """
        
        self.N_dim = 3 # by default, always.
        
        self.a, self.b, self.c, self.d = a, b, c, d
        self.e, self.f, self.tau = e, f, tau
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

        dv = 0.04*v**2 + 5*v + 140 - u + w_delay + I
        du = self.a * (self.b*v - u)
        dw = -1*self.e*w

        return np.array([dv, du, dw]) 
    



    def update_RK2(self, dt, N_iter, I_inj):
        """
        Simulates the neuron response to a injected current using RK2.
        
        IN:
            dt: timestep
            N_iter: number of iterations
            I_inj: array representing the injecter current [pA] of length N_iter
        OUT
            X: the membrane variables during the simulation
            T: corresponding time vector
            spike_times: array storing the time of spikes. 
        """


        # dt = timestep
        # N_iter = number of iterations
        # I_inj = injected current: len(I_inj) >= N_iter

        delay_steps = int(self.tau / dt)

        # Arrays to be returned, intially filled with all start values
        X = np.zeros((3, N_iter))
        X[:, 0] = self.x
    
        T = np.linspace(0, (N_iter - 1) * dt, N_iter)
        spike_times = []

        for n in range(1, N_iter): # can't start at 0 because this one is already calculated
            # Calculate the RK update variables.
            # Here, index n-1 corresponds to the 'current' value off which updates need to be estimated.

            w_tau1 = 0.0 if n <= delay_steps else X[2, n - delay_steps - 1]
            k1 = self.neuron_model(self.x, w_tau1, I_inj[n - 1])

            w_tau2 = self.x[2] + k1[2] * dt if delay_steps == 0 else X[2, max(n - delay_steps, 0)]
            k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[n])
            
            # Update x using RK2. Increment t.
            self.x += dt * (k1 + k2) / 2
            self.t = self.t + dt

            # Check for spike and reset variables.
            if self.x[0] >= 30:
                self.x[0] = self.c    # reset membrane potential
                self.x[1] += self.d    # bump membrane reset variable 
                self.x[2] += self.f    # bump autaptic variable
                spike_times.append(self.t)

            X[:, n] = self.x
        
        return X, T, np.array(spike_times)







