"""Simulates the model neuron response to various DC step currents
-  simulates both the neural response with and without
    an autapse present.

    Results are plotted for comparison.
 """

import numpy as np
import matplotlib.pyplot as plt
from AQUA2 import aqua2

def simulate_neuron():
    # RS parameters
    a = 0.02
    b = 0.2
    c = -65
    d = 8
    vpeak = 30  # spike cutoff

    T = 500  # total time (ms)
    h = 0.001  # time step (ms)

    # Set up plots
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))

    # Simulation for different current pulses
    for i, current in enumerate([5, 8, 11, 14]):
        I = current*np.ones(round(T/h))
        I[:int(0.1*round(T/h))] = 0.0
        # No autapse
        e = 0.0
        f = 0
        tau = 0
        v1, _, _, _ = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

        # With autapse
        e = 0.05 
        f = 6 + (i * 2)
        tau = 0
        v3, _, _, _ = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

        # Plotting
        axs[i, 0].plot(np.arange(len(I)) * h / 1000, I)
        axs[i, 0].set_ylim([min(I) - 1, 15])
        axs[i, 0].set_ylabel('I [pA]')
        axs[i, 0].set_xlabel('Time [sec]')
        axs[i, 0].set_title(f'Input Current: {current} pA')
        axs[i, 0].set_xticks([])

        axs[i, 1].plot(np.arange(len(v1)) * h / 1000, v1)
        axs[i, 1].set_ylabel('V [mV]')
        axs[i, 1].set_xlabel('Time [sec]')
        axs[i, 1].set_xticks([])

        axs[i, 2].plot(np.arange(len(v3)) * h / 1000, v3, color='orange')
        axs[i, 2].set_ylabel('V [mV]')
        axs[i, 2].set_xlabel('Time [sec]')
        axs[i, 2].set_xticks([])

        if i == 0:
            axs[i, 1].title.set_text('No autapse')
            axs[i, 2].title.set_text('With autapse')

    plt.tight_layout()
    plt.show()

simulate_neuron()