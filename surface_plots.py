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

from AQUA_general import AQUA
from batchAQUA_general import batchAQUA


"""
From https://scipython.com/blog/animated-landscapes-with-matplotib/

and

https://towardsdatascience.com/creating-a-gradient-descent-animation-in-python-3c4dcd20ca51-2/

make_Z = batchAQUA.neuron_model

"""
def make_Z(V, U, W, I, dt, batch):
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

    grad1 = batch.neuron_model(batch.x, 0., I)
    grad2 = batch.neuron_model(batch.x + dt*grad1, 0., I)

    grad = dt*(grad1 + grad2)/2.
    x_new = x_start + grad

    print(np.shape(x_new))
    print(np.shape(np.linalg.norm(x_new, axis = 1)))

    sign_Z = np.sign(np.linalg.norm(x_new, axis = 1) - np.linalg.norm(x_start, axis = 1))
    
    print("- - - -")
    print(np.shape(sign_Z))
    print(sign_Z[:10])

    #Z = sign_Z * np.linalg.norm(grad, axis = 1)
    #Z = np.reshape(Z, shape = np.shape(V))
    
    #return Z
    return np.reshape(sign_Z, shape = (np.shape(V)))


def make_Z_from_traj(X):
    """
        X: ndarray neuron trajectory (3, N_timesteps)
        I: injected current (N_timesteps, )
        W:
    """

    grad_vec = np.diff(X, axis = 1)
    print(np.shape(grad_vec))
    print(np.shape(np.linalg.norm(X, axis = 0)))
    print(np.shape(np.diff(np.linalg.norm(X, axis = 0))))
    sign_vec = np.sign(np.diff(np.linalg.norm(X, axis = 0)))
    print(np.shape(sign_vec))

    print(np.shape(np.linalg.norm(grad_vec, axis = 0)))

    return sign_vec * np.linalg.norm(grad_vec, axis = 0)


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

def draw_trajectory(x_ini, param, T, dt, I_h, fig, ax):
    """
    Draw the trajectory on a 3D surface.
    """

    N_iter = int(1000*T/dt)

    I_inj = I_h*np.ones(N_iter)

    t_start = np.array([0.])

    neuron = AQUA(param)
    neuron.Initialise(x_ini, t_start)

    X, _, _ = neuron.update_RK2(dt, N_iter, I_inj)

    Z = make_Z_from_traj(X)
    Z = np.append(Z, 0)

    new_axis = fig.add_axes(ax.get_position(), projection = '3d',
                            xlim = ax.get_xlim(),
                            ylim = ax.get_ylim(),
                            zlim = ax.get_zlim(),
                            facecolor = 'none',)
    new_axis.view_init(azim = ax.azim, elev = ax.elev)
    new_axis.set_zorder(1)
    ax.set_zorder(0)

    new_axis.plot3D(X[0, :], X[1, :], Z, color = 'red', alpha = 0.7)
    new_axis.plot3D(X[0, 0], X[1, 0], Z[0], ms = 3.5, c = 'red', marker = 'o')

    new_axis.set_axis_off()

    return fig, ax, new_axis

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
    
    ax.view_init(elev = 50, azim = 270)
    
    ani = animation.FuncAnimation(fig, animate, fargs = (surf,),
                                  interval=10, blit=True, frames = np.shape(I)[1])
                    
    if save:
        ani.save('phase_portrait_animation.mp4', fps = 24)
    
    return ani

