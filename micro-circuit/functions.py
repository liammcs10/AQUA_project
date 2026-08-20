import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

from matplotlib.colors import ListedColormap




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


def plot_heatmap(df, x_col, y_col, value_col, fixed_col, fixed_val, fig, ax, v_min=None, v_max=None):
    """
    Plots a heatmap for two variables while holding a third variable constant.
    - No limits: Symmetric 'vlag' centered at 0.
    - Limits provided: Sequential 'Reds' colormap.
    """
    # 1. Filter the dataframe
    filtered_df = df[np.isclose(df[fixed_col], fixed_val)]
    
    if filtered_df.empty:
        print(f"Warning: No data found where {fixed_col} == {fixed_val}")
        return ax

    # 2. Pivot the filtered data
    try:
        pivot_table = filtered_df.pivot(index=y_col, columns=x_col, values=value_col)
    except ValueError:
        pivot_table = filtered_df.pivot_table(index=y_col, columns=x_col, 
                                              values=value_col, aggfunc='mean')

    # 3. Handle Symmetry and Colormap Logic
    if v_min is None and v_max is None:
        # Symmetric mode
        limit = np.abs(pivot_table.values).max()
        v_min, v_max = -limit, limit
        cmap = "vlag"
        center = 0
    else:
        # Sequential "Red" Mode using only the second half of vlag
        vlag_full = plt.get_cmap("vlag")
        # Get the colors from the 50% mark to 100% mark
        red_half_colors = vlag_full(np.linspace(0.5, 1, 256))
        cmap = ListedColormap(red_half_colors)
        center = None

    # 4. Plot
    sns.heatmap(pivot_table, annot=False, cmap=cmap, ax=ax, 
                vmin=v_min, vmax=v_max, center=center)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    return ax

def create_heatmaps(df_aut, df_naut, diff_df, metric, fixed_values, v_min = None, v_max = None):
    # each row will be a different set of inputs
    fig, ax = plt.subplots(3, 3, figsize = (12, 8), sharex = 'col', sharey = 'row')
    fig.tight_layout()
    fig.suptitle(f'- - {metric} - -', x = 0.49, y = 1.0)

    # Injected vs w1
    plot_heatmap(df_aut, 'I_inj', 'w_e1_e2', metric, 'w_e2_e1', fixed_values[0], fig = fig, ax = ax[0, 0], v_min = v_min, v_max = v_max)
    plot_heatmap(df_naut, 'I_inj', 'w_e1_e2', metric, 'w_e2_e1', fixed_values[0], fig = fig, ax = ax[1, 0], v_min = v_min, v_max = v_max)
    plot_heatmap(diff_df, 'I_inj', 'w_e1_e2', metric, 'w_e2_e1', fixed_values[0], fig = fig, ax = ax[2, 0], v_min = None, v_max = None)

    # Injected vs w2
    plot_heatmap(df_aut, 'I_inj', 'w_e2_e1', metric, 'w_e1_e2', fixed_values[1], fig = fig, ax = ax[0, 1], v_min = v_min, v_max = v_max)
    plot_heatmap(df_naut, 'I_inj', 'w_e2_e1', metric, 'w_e1_e2', fixed_values[1], fig = fig, ax = ax[1, 1], v_min = v_min, v_max = v_max)
    plot_heatmap(diff_df, 'I_inj', 'w_e2_e1', metric, 'w_e1_e2', fixed_values[1], fig = fig, ax = ax[2, 1], v_min = None, v_max = None)

    # w1 vs w2
    plot_heatmap(df_aut, 'w_e1_e2', 'w_e2_e1', metric, 'I_inj', fixed_values[2], fig = fig, ax = ax[0, 2], v_min = v_min, v_max = v_max)
    plot_heatmap(df_naut, 'w_e1_e2', 'w_e2_e1', metric, 'I_inj', fixed_values[2], fig = fig, ax = ax[1, 2], v_min = v_min, v_max = v_max)
    plot_heatmap(diff_df, 'w_e1_e2', 'w_e2_e1', metric, 'I_inj', fixed_values[2], fig = fig, ax = ax[2, 2], v_min = None, v_max = None)

    rows = ["Autaptic Neuron", "Non-Autaptic Neuron", "Difference"]
    for axes, row_label in zip(ax[:,0], rows):
        axes.annotate(row_label, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - 5, 0),
                    xycoords=axes.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    
    return fig, ax


def plot_isi_peaks(spikes, bins, x_range, df, ax=None):
    """Plots the ISI histogram along with detected peaks, their vertical locations,

    and horizontal lines representing their width boundaries.

    Parameters:
    -----------
    counts : array-like
        The heights of the histogram bins.
    bin_edges : array-like
        The edges of the bins.
    peak_results : list of dicts
        The output from the `analyze_isi_peaks` function.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))


    ISI = np.diff(spikes)
    
    counts, bin_edges = np.histogram(ISI, bins, x_range)
    counts = gaussian_filter(counts, sigma = 2)

    # 1. Calculate bin centers and widths for plotting the histogram cleanly
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # 2. Plot the baseline ISI histogram
    ax.bar(
        bin_centers,
        counts,
        width=bin_width,
        color="skyblue",
        edgecolor="black",
        alpha=0.6,
        label="ISI Distribution",
    )

    for i, row in enumerate(df.itertuples()):

        # Extract values using the column names from your dataframe
        peak_val = row.peak_isi
        mean_val = row.mean
        std_val = row.std

        # Find the peak height in counts to anchor our vertical line
        peak_idx = np.argmin(np.abs(bin_centers - peak_val))
        peak_height = counts[peak_idx]

        # Draw vertical line at the peak location
        ax.axvline(
            x=peak_val,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label="Detected Peak" if i == 0 else "",
        )

        # Draw horizontal line representing the width (Mean ± STD)
        y_pos = peak_height * 0.2
        ax.hlines(
            y=y_pos,
            xmin=mean_val - std_val,
            xmax=mean_val + std_val,
            color="purple",
            linewidth=2.,
            label="Peak Width (Mean ± STD)" if i == 0 else "",
        )

        # Mark the actual weighted mean with a point
        '''
        ax.plot(
            mean_val,
            y_pos,
            "o",
            color="purple",
            markersize=8,
            label="Weighted Mean" if i == 0 else "",
        )
        '''

    ax.text(0., 3, s = entropy(counts))

    # Labels and aesthetics
    ax.set_xlabel("Inter-Spike Interval (ISI) [ms]")
    ax.set_ylabel("Counts")
    ax.set_title("ISI Distribution with Peak Analysis")
    #ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)


def plot_peak_and_series(f, idx_arr, spikes_E1, spikes_E2, X_E1, X_E2, df, f_vals, time):
    # plot ISI distributions for f1
    num_rows = len(f)

    layout = []

    for i in range(num_rows):
        # Generate unique names for the 3 axes in this specific row
        ax1 = f"R{i}_1"
        ax2 = f"R{i}_2"
        ax3 = f"R{i}_3"

        # Apply the 25% + 25% + 50% pattern
        row_pattern = [ax1, ax2, ax3, ax3]
        layout.append(row_pattern)


    fig, ax = plt.subplot_mosaic(layout, figsize = (15, 2*num_rows))
    bins = 300
    x_range = (0, 150)
    split = np.arange(40000, 50000)

    for n, idx in enumerate(idx_arr):

        peak_results_E1 = df[(df['f'] == f_vals[idx_arr[n]]) & (df['neuron_label'] == 'E1')]
        peak_results_E2 = df[(df['f'] == f_vals[idx_arr[n]]) & (df['neuron_label'] == 'E2')]
        plot_isi_peaks(spikes_E1[idx], bins = bins, x_range = x_range, df = peak_results_E1, ax = ax[layout[n][0]])
        plot_isi_peaks(spikes_E2[idx], bins = bins, x_range = x_range, df = peak_results_E2, ax = ax[layout[n][1]])
        
        # [spikes_E1[idx] > 1000]

        ax[layout[n][2]].plot(time[split], X_E1[idx, split], c = 'navy', label = 'E1')
        ax[layout[n][2]].plot(time[split], X_E2[idx, split], c = 'firebrick', label = 'E2')
        ax[layout[n][2]].legend()

        ax[layout[n][0]].set_ylabel(f'f = {f[n]}')
    
    return fig, ax