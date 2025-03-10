"""Simulates the model neuron response to a 14 pA dc-current
    simulates both the neural response with and without
    an autapse present.

    Results are plotted for comparison.
 """


import numpy as np
import matplotlib.pyplot as plt
from AQUA2 import aqua2


def simulate_neuron():
    a = 0.02
    b = 0.2
    c = -65
    d = 8
    vpeak = 30  # spike cutoff

    T = 500  # total time (ms)
    h = 0.001  # time step (ms)

    # Create a new figure
    plt.figure(figsize=(10, 8))

    # Input current
    I = 14*np.ones(round(T / h))
    I[:int(0.1*round(T/h))] = 0.0 # set first 10% of values to 0.0

    # No autapse
    e = 0.0; f = 0; tau = 0
    v1, _, _, _ = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

    # With autapse
    e = 0.05; f = 6; tau = 0
    v3, _, _, _ = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

    # Plotting
    time = np.arange(0, round(T / h)) * h / 1000  # Convert time to seconds

    plt.subplot(3, 1, 1)
    plt.plot(time, I)
    plt.xlim([0, 0.2])
    plt.ylim([min(I) - 1, 15])
    plt.ylabel('I [pA]')
    plt.title('Input Current')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(3, 1, 2)
    plt.plot(time, v1)
    plt.xlim([0, 0.2])
    plt.ylabel('V [mV]; no autapse')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(3, 1, 3)
    plt.plot(time, v3)
    plt.xlim([0, 0.2])
    plt.ylabel('V [mV]; with autapse')
    plt.xlabel('Time [sec]')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_neuron()