"""
Some functions for plotting/animating the phase portrait surface.

(1) plot surface (single timepoint) in UV-plane with height as update vector gradient
        - want color to show equilibria.
(2) Animate trajectory
(3) Animate surface
(4) Animate trajectory on moving surface

"""


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.colors import LightSource
import matplotlib.animation as animation
# create utils file?


from batchAQUA_general import batchAQUA


"""
From https://scipython.com/blog/animated-landscapes-with-matplotib/

and

https://towardsdatascience.com/creating-a-gradient-descent-animation-in-python-3c4dcd20ca51-2/

make_Z = batchAQUA.neuron_model

"""
def make_Z(V, U, W, I, batch):
    """
    Calculate the gradient vector for each point in meshgrid and normalise.
    """
    N_neurons = batch.N_models
    # first, initialise batch with mesh coords
    x_start = np.zeros((N_neurons, 3))
    x_start[:, 0] = V.flatten()
    x_start[:, 1] = U.flatten()
    x_start[:, 2] = W.flatten()

    t_start = np.zeros(N_neurons)

    batch.Initialise(x_start, t_start)

    grad = batch.neuron_model(batch.x, 0., I)

    Z = np.linalg.norm(grad, axis = 1)
    Z = np.reshape(Z, shape = np.shape(V))
    
    return Z

def plot_surface(ax, V, U, Z):
    """
    Plot a surface
    """

    cmap = plt.cm.terrain
    light = LightSource(90, 45)

    illuminated_surface = light.shade(Z, cmap = cmap)

    surf = ax.plot_surface(V, U, Z, rstride = 1, cstride = 1, lw = 0,
                            antialiased=False, facecolors = illuminated_surface)

    ax.set_xlim((np.min(V), np.max(V)))
    ax.set_ylim((np.min(U), np.max(U)))
    ax.axis('off')

    return surf



def make_animation(V, U, W, I, batch, plot_surf, save = False):
    """
    Make matplotlib animation
    """
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', facecolor='k'),
                           facecolor='k')

    NV, NU = V.shape[0], U.shape[0]
    Z = make_Z(V, U, W, I[:, 0], batch)      # create Z from initial conditions
    surf = plot_surf(ax, V, U, Z)
    ax.axis('off')

    def animate(i, surf):
        """animation function called for each animation frame"""
        ax.clear()
        Z = make_Z(V, U, W, I[:, i], batch)
        surf = plot_surf(ax, V, U, Z)
        return (surf,)

    ani = animation.FuncAnimation(fig, animate, fargs = (surf,),
                                  interval=10, blit=True, frames = 200)
                    
    if save:
        ani.save('phase_portrait_animation.mp4', fps = 24)
    
    return ani

