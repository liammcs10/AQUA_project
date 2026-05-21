import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

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