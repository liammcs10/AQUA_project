import numpy as np

class Hindmarsh_Rose:

    """ This class creates and evolves a single neuron according to the hindmarsh-rose model.
     Currently no autapse is implemented.
     Can also extend this code to allow for u, a, and b to be vectors.
    
    """

    # Set the model parameters. This will dictate the type of neuron 
    # being simulated
    def __init__(self, a, b, c, d, r, s, x_r):
        # a,b c, d, r, s x_r are HR params 
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.x_r = x_r


        self.x = []
        self.t = []



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
    def neuron_model(self, x, t, I):
        # x is 2D representing v and u
        # updates v and u according to the izhikevich model.
        # returns the gradients

        z = np.zeros([np.shape(self.x)[0]])

        z[0] = x[1] - self.a*x[0]**3 + self.b*x[0]**2 - x[2] + I       # x
        z[1] = self.c - self.d*x[0]**2 - x[1]                           # y
        z[2] = self.r*(self.s*(x[0] - self.x_r) - x[2])

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
            k1 = self.neuron_model(self.x, self.t, I[n-1])
            k2 = self.neuron_model(self.x+dt*k1, self.t+dt, I[n-1]) 

            # Update x using RK2. Increment t.
            self.x = self.x + dt*(k1 + k2)/2
            self.t = self.t + dt

            # Save values in the output arrays
            X[:, n] = np.copy(self.x)
            T[n] = np.copy(self.t)
        
        return X, T