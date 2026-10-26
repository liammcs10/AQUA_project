import numpy as np
import stumpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns




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


def changepoint_distribution():
    '''
    Compare different estimates of the change point for different autapse conditions
    
    '''
