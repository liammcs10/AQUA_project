

import aqua
from aqua.AQUA_general import AQUA
from aqua.batchAQUA_general import batchAQUA
from aqua.plotting_functions import *
from aqua.stimulus import *


import numpy as np
import matplotlib.pyplot as plt


""" - - - CONFIG PARSING HELPERS - - - """

def cast_to_float(data_dict):
    """
    Casts the values of a dictionary to float if conversion is possible.
    Otherwise, the original value is retained.
    """
    new_dict = {
        key: float(value) 
        if isinstance(value, (int, str)) and value not in ('', None) and is_float(value)
        else value
        for key, value in data_dict.items()
    }
    return new_dict

def is_float(value):
    """Helper function to safely check if a string can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


""" - - - PHASE PLOT FUNCTIONS - - - """


def calculate_nullclines(v, u, w, I, p):

    W = w*np.ones(len(v))
    I_inj = I*np.ones(len(v))

    v_null = (1/p["C"])*(p["k"]*(v - p["v_r"])*(v - p["v_t"]) + W + I_inj)
    u_null = p["a"]*(p["b"]*(v - p["v_r"]))

    return v_null, u_null



def plot_flow(U, V, batch, w, I, fig = None, ax = None):

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    N_neurons = batch.N_models
    # calculate dv, du
    deriv = batch.neuron_model(batch.x, w*np.ones(N_neurons), I*np.ones(N_neurons))

    dv = deriv[:, 0].reshape(V.shape)
    du = deriv[:, 1].reshape(U.shape)

    speed = np.sqrt(dv**2 + du**2)

    ax.streamplot(
                V,
                U,
                dv,
                du,
                color=speed,
                cmap="viridis",
                density=0.8,
                linewidth=0.5,
                arrowsize=1.0,
            )

    return fig, ax


def plot_nullclines(v_null, u_null, v, u, fig = None, ax = None):

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    linewidth = 2
    linestyle = '-'
    ax.plot(v, v_null, color = 'green', label = 'v nullcline', linestyle = linestyle, linewidth = linewidth)
    ax.plot(v, u_null, color = 'red', label = 'u nullcline', linestyle = linestyle, linewidth = linewidth)
    ax.set_xlim(np.min(v), np.max(v))
    ax.set_ylim(np.min(u), np.max(u))

    return fig, ax


def plot_trajectory(T, dt, I, params, x_start, fig, ax):


    N_iter = int(T/dt)
    # injected current
    I_inj = I*np.ones((N_iter))


    neuron = batchAQUA(params)

    t_start = np.array([0.])
    neuron.Initialise(x_start, t_start)

    X, T, spikes = neuron.update_batch(dt, N_iter, I_inj)

    split = np.arange(0, int(N_iter))

    collection = plot_time_gradient(X, T, split)
    ax.add_collection(collection)

    ax.scatter(x = X[:, 0, 0], y = X[:, 1, 0], color = 'black', marker = 's')

    return fig, ax

