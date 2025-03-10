"""
The GPT optimised version of the AQUA class.

batchAQUA_GPT uses numpy vectorization to simulate many neurons simultaneous
this can simulate 500 neurons for 2 seconds at 0.01ms resolution in ~26 seconds on my laptop.
"""



import numpy as np

class AQUA_GPT:
    def __init__(self, a, b, c, d, e=0.0, f=0.0, tau=0.0, E_syn = 10.0):
        self.N_dim = 3
        self.a, self.b, self.c, self.d = a, b, c, d
        self.e, self.f, self.tau = e, f, tau
        self.E_syn = E_syn
        self.x = np.zeros(3)
        self.t = 0.0

    def initialise(self, x_start, t_start):
        self.x[:] = x_start
        self.t = t_start #+ self.tau

    def neuron_model(self, x, w_delay, I):
        v, u, w = x
        dv = 0.04 * v**2 + 5 * v + 140 - u + w_delay + I
        du = self.a * (self.b * v - u)
        dw = -self.e * w
        return np.array([dv, du, dw])

    def update_RK2(self, dt, N_iter, I_inj):
        delay_steps = int(self.tau / dt)
        X = np.zeros((3, N_iter))
        X[:, 0] = self.x
        T = np.linspace(0, (N_iter - 1) * dt, N_iter)
        spike_times = []

        for n in range(1, N_iter):
            w_tau1 = 0.0 if n <= delay_steps else X[2, n - delay_steps - 1]
            k1 = self.neuron_model(self.x, w_tau1, I_inj[n - 1])

            w_tau2 = self.x[2] + k1[2] * dt if delay_steps == 0 else X[2, max(n - delay_steps, 0)]
            k2 = self.neuron_model(self.x + dt * k1, w_tau2, I_inj[n])

            self.x += dt * (k1 + k2) / 2
            self.t = self.t + dt

            if self.x[0] >= 30:
                self.x[0] = self.c
                self.x[1] += self.d
                self.x[2] += (self.f/self.E_syn) * (self.E_syn - self.x[2])
                spike_times.append(self.t)

            X[:, n] = self.x

        return X, T, np.array(spike_times)
    



