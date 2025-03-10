import numpy as np

class Neuron:

    """ This class creates and evolves a single neuron according to the izhikevich model.
     Currently no autapse is implemented.
     Can also extend this code to allow for u, a, and b to be vectors.
    
    """

    # Set the model parameters. This will dictate the type of neuron 
    # being simulated
    def __init__(self, a, b, c, d):
        # a,b c, d are izhikevich params for cortical neurons
        self.a = a
        self.b = b
        self.c = c
        self.d = d


        self.x = []
        self.t = []
        self.spike_time = [] # will store the times of a spike



    # Set starting values of the variables for the simulation
    def Initialise(self, x_start, t_start):
        # Set starting values
        # x is 2D array [v, u]
        # t is a float.
        self.x = x_start 
        self.t = t_start

        self.N_dim = np.shape(x_start)[0] # extract the number of dimensions.



    # Izhikevich differential equations for the membrane potential v and 
    # the recovery variable u.
    def neuron_model(self, x, t, I, a, b):
        # x is 2D representing v and u
        # updates v and u according to the izhikevich model.
        # returns the gradients

        z = np.zeros([np.shape(self.x)[0]])

        z[0] = 0.04*x[0]**2 + 5*x[0] + 140 - x[1] + I
        z[1] = a*(b*x[0] - x[1])

        return z 
    


    # Setup and update the neuron variables in response to an input current
    # This method uses RK2 to simulate neural responses.
    # I_inj needs to be known ahead of time....
    def update_RK2(self, dt, N_iter, I_inj):
        # dt = timestep
        # N_iter = number of iterations
        # I_inj = injected current: len(I_inj)>= N_iter

        X = np.zeros([self.N_dim, N_iter])
        T = np.zeros([N_iter])

        #injected current, 
        I = I_inj

        # Initial values
        X[:, 0] = np.copy(self.x)
        T[0] = np.copy(self.t)

        for n in range(1, N_iter): # can't start at 0 because this one is already calculated
            
            # Calculate the RK update variables.
            # For k2 can we use the updated current?
            k1 = self.neuron_model(self.x, self.t, I[n-1], self.a, self.b)
            k2 = self.neuron_model(self.x+dt*k1, self.t+dt, I[n-1], self.a, self.b) 

            # Update x using RK2. Increment t.
            self.x = self.x + dt*(k1 + k2)/2
            self.t = self.t + dt

            # Check for spike and reset variables.
            if self.x[0] >= 30:
                self.x[0] = self.c
                self.x[1] = self.x[1] + self.d
                #record the spike.
                self.spike_time = np.append(self.spike_time, self.t)

            # Save values in the output arrays
            X[:, n] = np.copy(self.x)
            T[n] = np.copy(self.t)
        
        return X, T, self.spike_time








