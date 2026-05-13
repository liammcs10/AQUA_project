"""
Here we will try to implement a simple change point analysis protocol using the AQUA model.

Injected current will contain 1 change point at a fixed time where the mean of the injected current will change. There will be a super imposed noise.

The question: Based on the spiking activity alone, is it possible to estimate the change point? How significant does the change point have to be? 

** Can the autapse help distinguish smaller changes in activity?

- What happens if multiple change points are present? 

PARAMETERS TO VARY:
- The change in the mean (** CAN ALSO ADJUST THE STARTING CURRENT VALUE **)
- Variance of the noise (** WE CAN ALSO VARY THE TYPE OF NOISE **)
- autapse parameters
- window size of the fluss_detector (How does it fare with more or less data to compare...?)


OUTPUTS:

Psychometric curve - does the autapse shift the curve to the left or right?

Distribution of change point times - Does the autapse give a better estimate overall?

"""
import warnings
warnings.filterwarnings("ignore", module = "stumpy")



import aqua

from aqua.batchAQUA_general import batchAQUA
from aqua.utils import *
from aqua.stimulus import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.signal import convolve, windows
from tqdm import tqdm
import pickle
import seaborn as sns
sns.set_theme()


from changepoynt.algorithms.fluss import FLUSS
from changepoynt.visualization.score_plotting import plot_data_and_score


# neuron model (RS integrator)
RS = {'name': 'RS_ref', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}



def main():
    '''
    Run the estimate on many different parameter combinations.
    
    Parameters to vary : autapse (can make it narrower than that in other sims), height of change point, noise,
    window of the FLUSS analyser.

    We can also compare values for the raw spiking time series and also the ISI distribution.  

    '''

    # for now only vary the peak autapse current? 
    f_values = np.linspace(100, 350, 15)

    # height of the change point
    #d_H = np.linspace(0, 50, 10)
    d_H = np.linspace(0, 5, 10)

    # noise variance 
    #var = np.linspace(0, 50, 5)
    var = np.array([0])

    # FLUSS window size (in time steps)
    window = 300
    print(f"WINDOW: {window}")

    outfile = 'CPA_test_.pickle'
    simulate(RS, f_values, d_H, var, window, outfile)






def simulate(model, f_values, d_H, var, window, outfile):
    """
    Initialise and run a large network with different parameter configurations. 
    Saves data directly to then be imported and processed by the other functions.
    
    """

    T = 3000 # ms
    dt = 0.1
    N_iter = int(T/dt)

    # create parameter dictionary
    params = []
    params.append(model)    # add the non-autaptic neuron
    e = 0.2     # /ms
    tau = 2.0   # ms

    for f in f_values:
        temp_dict = model.copy()
        temp_dict['e'] = e
        temp_dict['tau'] = tau
        temp_dict['f'] = np.round(f, 2)
        params.append(temp_dict)

    N_neurons = len(params)
    print(f"N_neurons: {N_neurons}")

    # Create the different changepoint currents. The changepoint will be exactly at the halfway point
    N_sims = N_neurons * len(d_H) * len(var)
    print(f"N_SIMS: {N_sims}")

    I_0 = 60    # pA, initial driving current
    changepoint = 1500 # ms
    I_inj = np.zeros((N_sims, N_iter))
    N_p_var = N_neurons * len(d_H)  # number of iterations per var
    sim_params = []
    for l, v in enumerate(var):
        for m, h in enumerate(d_H):
            step = np.array([step_current(N_iter, dt, y_0 = I_0, delay = changepoint, I_h = I_0 + h) for i in range(N_neurons)])
            WN = np.random.normal(loc = 0.0, scale = np.sqrt(v), size = (N_iter))
            step_w_noise = step + WN
            idx = l * N_p_var + m * N_neurons
            I_inj[idx:idx+N_neurons, :] = step_w_noise
            sim_params.append(params)
    sim_params = list(np.array(sim_params).flatten())
    print(sim_params[:5])
    params_df = pd.DataFrame(sim_params)
    
    print(f"SIM_PARAMS: {len(sim_params)}")

    print(params_df.head())

    # start values        
    x_start = np.full((N_sims, 3), fill_value = np.array([-60, 0, 0]))
    t_start = np.zeros(N_sims)


    # create batch and initialise
    batch = batchAQUA(params_df)
    batch.Initialise(x_start, t_start)

    # simulate
    X, t, spikes =  batch.update_batch(dt, N_iter, I_inj)


    # Now need to process the outputs using the FLUSS algorithm...
    '''
    Target output:
        DataFrame:  'var', 'H', 'f', 'window', 'peak CAC', 'estimated CP'
    
    Analysis is re-run for each window size (outer loop)


    '''
    f_values = np.insert(f_values, 0, 0.)
    cols = ['ISI_TS', 'variance', 'step_height', 'autapse f', 'FLUSS window', 'CAC', 'change point']
    data_df = pd.DataFrame(data = [], columns = cols)

    ISI_trace = get_ISI_time_series(spikes, N_iter, dt)

    row = 0

    # initialise the analyser
    fluss = FLUSS(window_length = window)

    for v in tqdm(var):
        for h in d_H:
            for f in f_values:

                # Run analysis on membrane potential
                score = fluss.transform(X[row, 0, :])   # fluss on membrane potential
                CAC = np.max(score)
                change_time = np.argmax(score)*dt   # in ms
                df = pd.DataFrame(data = [[False, v, h, f, window, CAC, change_time]], columns = cols)
                data_df = pd.concat([data_df, df], axis = 0, ignore_index = True)
                
                # Run on the ISI time series
                score_isi = fluss.transform(ISI_trace[row, :])
                CAC_isi = np.max(score_isi)
                change_time_isi = np.argmax(score_isi)*dt
                df_isi = pd.DataFrame(data = [[True, v, h, f, window, CAC_isi, change_time_isi]], columns = cols)
                data_df = pd.concat([data_df, df_isi], axis = 0, ignore_index = True)

                row += 1
                
    print(data_df.head())

    # save the dataframe
    with open(outfile, 'wb') as file:
        pickle.dump(data_df, file)





def psychometric_curve(data, x, y, hue = 'autapse f', fig = None, ax = None, add_colorbar = True):
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
    sns.lineplot(data=data, x=x, y=y, hue=hue, palette=cmap_name, ax = ax, legend = False)
    #ax.set_ylim((0, 1))

    if add_colorbar:
        # 3. Create the Colorbar
        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])  # Dummy array for the mappable

        # Add the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Autapse Strength')

    return fig, ax


def changepoint_distribution():
    '''
    Compare different estimates of the change point for different autapse conditions
    
    '''


if __name__ == "__main__":
    main()