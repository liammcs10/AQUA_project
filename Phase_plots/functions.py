"""
Phase-plot helper functions for the AQUA RS autaptic-neuron project.

These follow the style set out in testing.ipynb: the flow field is computed
over a V-U grid with a *batch* of AQUA neurons (one neuron per grid point),
and trajectories are drawn as time-gradient line collections
(aqua.plotting_functions.plot_time_gradient) rather than plain lines.
"""

import sys
import types

# batchAQUA_general (via aqua.utils) imports brian2 unconditionally, even
# though nothing here uses brian2. Stub it out so this module works in
# environments where brian2 isn't installed.
if "brian2" not in sys.modules:
    try:
        import brian2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["brian2"] = types.ModuleType("brian2")

import numpy as np
import matplotlib.pyplot as plt

from aqua.batchAQUA_general import batchAQUA
from aqua.plotting_functions import plot_time_gradient, plot_membrane_variables


""" - - - CONFIG PARSING HELPERS - - - """

def is_float(value):
    """Helper function to safely check if a string can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def cast_to_float(data_dict):
    """
    Casts the values of a dictionary to float if conversion is possible.
    Otherwise, the original value is retained.
    """
    return {
        key: float(value)
        if isinstance(value, (int, str)) and value not in ('', None) and is_float(value)
        else value
        for key, value in data_dict.items()
    }


def parse_bool(value, default=True):
    """Interpret a config string as a boolean, falling back to `default`."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


""" - - - NEURON PARAMETER HELPERS - - - """

def no_autapse_variant(params):
    """Return a copy of `params` with the autaptic current disabled (e = f = tau = 0)."""
    variant = dict(params)
    variant["e"] = 0.0
    variant["f"] = 0.0
    variant["tau"] = 0.0
    variant["name"] = f"{params['name']}_no_autapse"
    return variant


""" - - - SIMULATION - - - """

def simulate_trajectory(params, T, dt, I_inj_value, x_start, t_start=0.0):
    """
    Simulate a single neuron forward under a constant injected current using
    batchAQUA (a batch of size 1), matching the batch-oriented style used
    for the flow-field grid.

    Returns
    -------
    x_trace :   (3, N_iter) array of [v, u, w] through the simulation
    t_trace :   (N_iter,) array of time values
    spikes :    (N_spikes,) array of spike times (NaN padding stripped)
    """
    n_iter = int(T / dt)

    i_trace = I_inj_value * np.ones((1, n_iter))

    neuron = batchAQUA([params])
    neuron.Initialise(np.array([x_start], dtype=float), np.array([t_start], dtype=float))
    X, t_trace, spikes = neuron.update_batch(dt, n_iter, i_trace)

    spike_row = np.asarray(spikes[0], dtype=float)
    spike_row = spike_row[~np.isnan(spike_row)]

    if len(spike_row) == 0:
        raise RuntimeError(
            f"No spikes were produced for neuron '{params['name']}'. "
            "Increase I_inj or T in the [Simulation] section of the config."
        )

    return X[0], t_trace, spike_row


""" - - - PHASE PLOT FUNCTIONS - - - """

def calculate_nullclines(v, w, I, p):
    """v- and u-nullclines (dv/dt = 0, du/dt = 0) for a 1D array of v values."""
    W = w * np.ones(len(v))
    I_inj = I * np.ones(len(v))

    v_null = (1 / p["C"]) * (p["k"] * (v - p["v_r"]) * (v - p["v_t"]) + W + I_inj)
    u_null = p["a"] * (p["b"] * (v - p["v_r"]))

    return v_null, u_null


def build_grid_batch(params, v_grid):
    """Build a batchAQUA batch with one neuron per V-U grid point, reused
    across snapshots since the neuron parameters never change (only the
    frozen autapse current `w` does)."""
    n_points = v_grid.size
    batch = batchAQUA([params] * n_points)
    return batch


def compute_flow_field(grid_batch, v_grid, u_grid, w, I):
    """dv/dt, du/dt across the V-U grid for a frozen autapse current `w`."""
    n_points = v_grid.size
    x = np.column_stack(
        [v_grid.ravel(), u_grid.ravel(), np.full(n_points, w, dtype=float)]
    )
    w_arr = np.full(n_points, w, dtype=float)
    i_arr = np.full(n_points, I, dtype=float)

    deriv = grid_batch.neuron_model(x, w_arr, i_arr)

    dv = deriv[:, 0].reshape(v_grid.shape)
    du = deriv[:, 1].reshape(v_grid.shape)
    return dv, du


def plot_flow(v_grid, u_grid, dv, du, fig=None, ax=None, density=0.8, linewidth=0.5, cmap="viridis"):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    speed = np.sqrt(dv ** 2 + du ** 2)

    ax.streamplot(
        v_grid,
        u_grid,
        dv,
        du,
        color=speed,
        cmap=cmap,
        density=density,
        linewidth=linewidth,
        arrowsize=1.0,
    )

    return fig, ax


def plot_nullclines(v_null, u_null, v, u, fig=None, ax=None):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(v, v_null, color="green", label="v nullcline", linestyle="-", linewidth=2)
    ax.plot(v, u_null, color="red", label="u nullcline", linestyle="-", linewidth=2)
    ax.set_xlim(np.min(v), np.max(v))
    ax.set_ylim(np.min(u), np.max(u))

    return fig, ax


def plot_trajectory(x_trace, t_trace, split, fig=None, ax=None):
    """Draw a trajectory window as a time-gradient line, in the style of
    testing.ipynb (plot_time_gradient + a square marker at the start)."""
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    collection = plot_time_gradient(x_trace, t_trace, split)
    collection.set_label("trajectory")
    ax.add_collection(collection)
    ax.scatter(x_trace[0, split[0]], x_trace[1, split[0]], color="black", marker="s", zorder=4)

    return fig, ax


""" - - - EVENT-RELATIVE SNAPSHOT SELECTION - - - """

def index_at_time(t_trace, target_time):
    return int(np.clip(np.searchsorted(t_trace, target_time), 0, len(t_trace) - 1))


def choose_snapshot_indices(
    t_trace,
    x_trace,
    spikes,
    tau,
    dt,
    pre_spike_offset_ms,
    post_spike_offset_ms,
    immediate_autapse_offset_ms,
    late_autapse_offset_ms,
    autapse_died_down_fraction,
):
    """
    Choose five event-relative snapshots around the first spike: just before
    the spike, after the spike but before delayed feedback, immediately
    after delayed feedback, later during autapse decay, and either when the
    delivered autapse current has almost fully decayed or the latest
    possible time before the second spike/reset.
    """
    first_spike_time = spikes[0]
    if len(spikes) >= 2:
        latest_snapshot_time = spikes[1] - pre_spike_offset_ms
    else:
        latest_snapshot_time = t_trace[-1]

    autapse_start_time = first_spike_time + tau

    if tau <= dt:
        raise RuntimeError(
            "The configured tau is too short to show a post-spike/pre-autapse panel. "
            "Increase tau or reduce dt in the config."
        )

    post_spike_offset = min(post_spike_offset_ms, max(dt, 0.5 * tau))
    autapse_start_idx = index_at_time(t_trace, autapse_start_time)
    w_trace = x_trace[2]
    latest_snapshot_idx = index_at_time(t_trace, latest_snapshot_time)
    active_w = w_trace[autapse_start_idx : latest_snapshot_idx + 1]
    peak_w = np.max(active_w)

    if peak_w <= 0.0:
        raise RuntimeError(
            "No delivered autapse current was found. Check e, f, tau, dt, and T."
        )

    died_down_threshold = autapse_died_down_fraction * peak_w
    post_peak = active_w[np.argmax(active_w):]
    died_down_after_peak = np.flatnonzero(post_peak <= died_down_threshold)
    if len(died_down_after_peak) == 0:
        died_down_idx = latest_snapshot_idx
        final_snapshot_label = "latest before second spike"
    else:
        died_down_idx = autapse_start_idx + np.argmax(active_w) + died_down_after_peak[0]
        final_snapshot_label = "autapse died down"

    died_down_idx = min(died_down_idx, latest_snapshot_idx)

    snapshot_times = np.array(
        [
            first_spike_time - pre_spike_offset_ms,
            first_spike_time + post_spike_offset,
            autapse_start_time + immediate_autapse_offset_ms,
            autapse_start_time + late_autapse_offset_ms,
            t_trace[died_down_idx],
        ]
    )
    snapshot_times[1:] = np.minimum(snapshot_times[1:], latest_snapshot_time)

    snapshot_labels = [
        "just before spike",
        "after spike, before autapse",
        "autapse onset",
        "later autapse decay",
        final_snapshot_label,
    ]
    snapshot_indices = np.array([index_at_time(t_trace, t) for t in snapshot_times])
    return snapshot_indices, snapshot_labels


def compute_trajectory_window(
    t_trace, spikes, snapshot_indices, dt, trajectory_window_ms, trajectory_cutoff_ms=None
):
    """
    Determine the [traj_start, traj_stop) index window used to draw the
    trajectory in each phase-plot panel.

    By default the window starts a little before the first snapshot and runs
    until the second spike (or, if there is no second spike,
    `trajectory_window_ms` past the last snapshot). `trajectory_cutoff_ms`,
    if given and positive, additionally caps the total duration shown,
    measured from the start of the window, letting the displayed trajectory
    be shortened independently of where the next spike happens to fall.
    """
    pad = int(8.0 / dt)
    traj_start = max(snapshot_indices[0] - pad, 0)
    default_traj_stop = min(
        snapshot_indices[-1] + int(trajectory_window_ms / dt),
        len(t_trace) - 1,
    )
    if len(spikes) >= 2:
        traj_stop = index_at_time(t_trace, spikes[1])
    else:
        traj_stop = default_traj_stop

    if trajectory_cutoff_ms is not None and trajectory_cutoff_ms > 0:
        traj_stop = min(traj_stop, traj_start + int(trajectory_cutoff_ms / dt))

    traj_stop = min(traj_stop, len(t_trace) - 1)
    traj_stop = max(traj_stop, traj_start + 1)
    return traj_start, traj_stop


""" - - - COMPOSITE FIGURES - - - """

def plot_phase_grid(
    row_specs,
    snapshot_indices,
    snapshot_labels,
    v_grid,
    u_grid,
    v_lin,
    u_lin,
    I_inj,
    v_min,
    v_max,
    u_min,
    u_max,
    show_nullclines,
    density,
    linewidth,
    suptitle,
):
    """
    row_specs: list of dicts, one per row, each with keys:
        label, params, grid_batch, x_trace, t_trace, traj_start, traj_stop
    """
    n_cols = len(snapshot_indices)
    n_rows = len(row_specs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.3 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)

    for row_idx, row in enumerate(row_specs):
        params = row["params"]
        x_trace = row["x_trace"]
        t_trace = row["t_trace"]
        grid_batch = row["grid_batch"]
        traj_start, traj_stop = row["traj_start"], row["traj_stop"]
        split = np.arange(traj_start, traj_stop)

        for col_idx, (idx, label) in enumerate(zip(snapshot_indices, snapshot_labels)):
            ax = axes[row_idx, col_idx]
            w_delivered = x_trace[2, idx]

            dv, du = compute_flow_field(grid_batch, v_grid, u_grid, w_delivered, I_inj)
            plot_flow(v_grid, u_grid, dv, du, fig=fig, ax=ax, density=density, linewidth=linewidth)

            if show_nullclines:
                v_null, u_null = calculate_nullclines(v_lin, w_delivered, I_inj, params)
                plot_nullclines(v_null, u_null, v_lin, u_lin, fig=fig, ax=ax)

            collection = plot_time_gradient(x_trace, t_trace, split)
            collection.set_label("trajectory")
            ax.add_collection(collection)

            ax.scatter(
                x_trace[0, idx],
                x_trace[1, idx],
                color="crimson",
                s=35,
                zorder=5,
                label="snapshot",
            )
            ax.axvline(params["v_peak"], color="0.35", lw=0.9, ls=":")
            ax.set_title(
                f"{label}\n{row['label']}: t = {t_trace[idx]:.2f} ms, "
                f"w_delay = {w_delivered:.2f} pA",
                fontsize=9,
            )
            ax.set_xlim(v_min, v_max)
            ax.set_ylim(u_min, u_max)
            if row_idx == n_rows - 1:
                ax.set_xlabel("v (mV)")

        axes[row_idx, 0].set_ylabel(f"{row['label']}\nu")

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    return fig


def plot_time_series(x_trace, t_trace, spikes, snapshot_indices, snapshot_labels, title):
    fig, axes = plot_membrane_variables(x_trace, t_trace)

    for ax in axes:
        for spike_time in spikes:
            ax.axvline(spike_time, color="0.2", lw=0.8, ls=":", alpha=0.7)
        for idx in snapshot_indices:
            ax.axvline(t_trace[idx], color="crimson", lw=0.9, alpha=0.75)

    for idx, label in zip(snapshot_indices, snapshot_labels):
        axes[0].annotate(
            label,
            xy=(t_trace[idx], x_trace[0, idx]),
            xytext=(4, 8),
            textcoords="offset points",
            rotation=25,
            fontsize=8,
            color="crimson",
        )

    fig.suptitle(title)
    fig.tight_layout()
    
    return fig
