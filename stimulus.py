""" implement a class which will contain standard injected current traces for each neuron

e.g.
    Ornstein-Uhlenbeck process
    Poisson spike trains
    ...


"""
import numpy as np

"""
def __init__(self, N_iter, dt, y_0):

    self.N_iter = N_iter
    self.dt = dt
    self.y_0 = y_0
"""

def OU_current(N_iter, dt, y_0, theta, mu, sigma):
    # implemented using the Euler-Maruyama method.
    ys = np.zeros(N_iter)
    ys[0] = y_0
    for i in range(1, N_iter):
        y = ys[i-1]
        dW = np.random.normal(loc = 0.0, scale = np.sqrt(dt))
        ys[i] = y + theta * (mu - y) * dt + sigma * dW
    
    return ys

def ramp(N_iter, dt, y_0, delay, I_end):
    """
    Applies a ramp current starting at delay
    IN
        delay: the delay [ms] before applying the ramp
        I_end: The final current value

    OUT
        I_inj: np.array representing the ramp stimulus
    """
    delay_steps = int(delay / dt)       # the delay in units of timesteps.

    return np.concatenate((y_0 * np.ones(delay_steps), np.linspace(0, I_end, (N_iter - delay_steps))))


def step_current(N_iter, dt, y_0, delay, I_h):
    """
    A step current of heigh I_h applied at delay
    IN
        delay: the delay [ms] before the step
        I_h: the height of the step [pA]
    OUT
        I_inj: np.array of length N_iter containing the stimulus trace    
    """
    delay_steps = int(delay / dt)
    
    return np.concatenate((y_0 * np.ones(delay_steps), I_h * np.ones(N_iter - delay_steps)))
