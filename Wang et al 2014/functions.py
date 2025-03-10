## This file contains the plotting functions used by main.py


import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt



def plot_heatmap(df, label1, label2, name):
    """
    Plots a heat map of the information entropy of neuron responses
    as a function of the e and f autaptic parameters.

    Parameters
    ----------
    dict :      DataFrame of floats
                contains the simulations variables for each run including
                cols: [I_ext, e, f, tau, ISI_values, ISI_counts]

    Output
    ----------
    heatmap :   fig, ax
                Heatmap of information entropy
    
    """
    label1_vals = df[label1].unique() # x
    label2_vals = df[label2].unique() # y
    N_1 = len(label1_vals)
    N_2 = len(label2_vals)
    
    H_isi = np.zeros((N_2, N_1))

    for n, x in enumerate(label1_vals):
        for m, y in enumerate(label2_vals):
            counts = df[(df[label1] == x) & (df[label2] == y)]['ISI_counts']
            probs = counts/np.sum(counts)
            H_isi[m, n] = entropy(probs)
    
    fig, ax = plt.subplots(figsize = (7, 7))
    plt.imshow(H_isi, origin = 'lower')
    plt.xticks(range(N_1), labels = np.round(label1_vals, decimals = 2))
    plt.yticks(range(N_2), labels = np.round(label2_vals, decimals = 2))
    plt.xlabel(f'{label1} parameter values')
    plt.ylabel(f'{label2} parameter values')
    plt.colorbar()
    plt.savefig(f'{name}_{label1}-{label2}_map.png')
