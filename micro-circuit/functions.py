import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def plot_ISI_w_peaks(spike_times, bins = 50, x_range = (0, 100), fig = None, ax = None):
    # --- 1. Calculate ISIs ---
    # Assuming spike_times is your 1D array of spike timestamps
    isis = np.diff(spike_times)

    # --- 2. Create Histogram ---
    # You can adjust 'bins' to change the resolution of your analysis
    counts, bin_edges = np.histogram(isis, bins = bins, range = x_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 3. Detect Peaks and Widths ---
    # height: minimum count to be considered a peak
    # distance: minimum number of bins between peaks
    peaks, properties = find_peaks(counts, height=np.max(counts)*0.1, distance=5)
    results_half = peak_widths(counts, peaks, rel_height=0.5)

    # Mapping indices back to time units for plotting
    def idx_to_val(idx):
        return np.interp(idx, np.arange(len(bin_centers)), bin_centers)

    peak_times = bin_centers[peaks]
    left_vals = idx_to_val(results_half[2])
    right_vals = idx_to_val(results_half[3])
    width_heights = results_half[1]

    # --- 4. Plotting ---
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    
    ax.hist(isis, bins=50, range = x_range, color='skyblue', edgecolor='black', alpha=0.7, label='ISI Histogram')

    # Add vertical lines at peaks
    for pt in peak_times:
        ax.axvline(x=pt, color='red', linestyle='--', linewidth=2, label='Peak' if pt == peak_times[0] else "")

    # Add horizontal double-headed arrows for widths
    for i in range(len(peaks)):
        y = width_heights[i]
        ax.annotate('', xy=(left_vals[i], y), xytext=(right_vals[i], y),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2, shrinkA=0, shrinkB=0))
        ax.text((left_vals[i] + right_vals[i])/2, y, f' {right_vals[i]-left_vals[i]:.3f} ms', 
                ha='center', va='bottom', color='green', fontweight='bold')

    ax.set_xlabel('Interspike Interval (ms)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_range)


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')