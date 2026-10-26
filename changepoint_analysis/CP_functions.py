import numpy as np
import stumpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns


def find_idx(f, v, h, f_vals, v_vals, h_vals):
    """finds the index of a given set of parameters values
    given the list of those parameter values which
    defined the population simulations.
    Params:
        f, v, h:    target parameter values
        *_vals:     reference arrays in which to find targets
    
    """
    N_f = len(f_vals)
    N_v = len(v_vals)
    N_h = len(h_vals)

    idx_f = np.argwhere(f_vals == f)[0][0]
    idx_v = np.argwhere(v_vals == v)[0][0]
    idx_h = np.argwhere(h_vals == h)[0][0]

    idx = idx_v * (N_f*N_h) + idx_h * N_f + idx_f

    return int(idx)


def process_single_row(row_idx, data_matrix, window_size, dt):
    '''
    Computes the matrix profile on a single row and returns the CAC
    '''
    row_data = data_matrix[row_idx, 0, :]

    mat_prof = stumpy.stump(row_data, m = window_size)

    arc_crossings, _ = stumpy.fluss(mat_prof[:, 1], L = window_size, n_regimes = 1)

    score = 1 - arc_crossings

    CAC = np.max(score)
    CP = np.argmax(score)*dt

    return (CAC, CP)


def psychometric_curve(data, x, y, hue = 'autapse f', fig = None, ax = None, add_colorbar = True, colorbar_label = 'Autapse Strength'):
    '''
    Plot the psychometric curve for different autapse conditions
    
    '''
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    # 1. Define your colormap
    cmap_name = "viridis"
    norm = mcolors.Normalize(vmin=data[hue].min(), vmax=data[hue].max())

    #palette = {val: color for val, color in zip([s for s in autapses if s != '0.0'], default_colors)}
    #palette['0.0'] = 'black'  # Your pre-defined color

    # 3. Plot
    sns.lineplot(data=data, x=x, y=y, hue=hue, palette=cmap_name, ax = ax, legend = False, 
                 estimator = 'mean', errorbar = 'sd', err_style = 'band', alpha = 0.8)
    #ax.set_ylim((0, 1))

    if add_colorbar:
        # 3. Create the Colorbar
        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])  # Dummy array for the mappable

        # Add the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(colorbar_label)

    return fig, ax


def changepoint_distribution(data, x = 'change point', row = 'autapse f', col = 'variance', fig = None, ax = None,
                             bins = 60, true_cp = 1500, add_colorbar = True, colorbar_label = 'Autapse Strength'):
    '''
    Grid of histograms of the estimated change point times. One row per value of `row`
    (autapse strength), one column per value of `col` (noise variance). Histograms are
    coloured by the row value. Pools all step heights and repeats: filter `data` first
    to look at a single step height.

    Returns fig, ax (2D array of axes).
    '''
    row_vals = np.sort(data[row].unique())
    col_vals = np.sort(data[col].unique())

    if fig is None or ax is None:
        fig, ax = plt.subplots(len(row_vals), len(col_vals), figsize = (3*len(col_vals), 1.6*len(row_vals)),
                               sharex = True, sharey = True, squeeze = False)
    ax = np.atleast_2d(ax)

    cmap_name = "viridis"
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin = data[row].min(), vmax = data[row].max())

    # same bins for every panel
    edges = np.histogram_bin_edges(data[x], bins = bins)

    for i, r in enumerate(row_vals):
        for j, c in enumerate(col_vals):
            sub = data[(data[row] == r) & (data[col] == c)]
            sns.histplot(sub[x], bins = edges, color = cmap(norm(r)), ax = ax[i, j])

            if true_cp is not None:
                ax[i, j].axvline(true_cp, color = 'red', linestyle = '--', linewidth = 1)

            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            if i == 0:
                ax[i, j].set_title(f'{col.capitalize()} = {c}')
            if j == 0:
                ax[i, j].set_ylabel(f'f = {r} pA')
            if i == len(row_vals) - 1:
                ax[i, j].set_xlabel('Change point [ms]')

    if add_colorbar:
        sm = cm.ScalarMappable(cmap = cmap_name, norm = norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = ax)
        cbar.set_label(colorbar_label)

    return fig, ax
