import sys
import types
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# AQUA / RS NEURON PARAMETERS
# Values match the RS integrator model used in ../bursting/RS_int_uniform_config.ini
# ---------------------------------------------------------------------------
AUTAPTIC_NEURON_PARAMS = {
    "name": "RS_intHD",
    "C": 100.0,
    "k": 0.7,
    "v_r": -60.0,
    "v_t": -40.0,
    "v_peak": 35.0,
    "a": 0.03,
    "b": -2.0,
    "c": -50.0,
    "d": 100.0,
    "e": 0.2,
    "f": 150.0,
    "tau": 2.0,
}

NON_AUTAPTIC_NEURON_PARAMS = AUTAPTIC_NEURON_PARAMS.copy()
NON_AUTAPTIC_NEURON_PARAMS.update(
    {
        "name": "RS_intHD_no_autapse",
        "e": 0.0,
        "f": 0.0,
        "tau": 0.0,
    }
)

# ---------------------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------------------
DT = 0.05
T_MAX = 350.0
I_INJ = 100.0
X_START = np.array([-60.0, 0.0, 0.0])
T_START = 0.0

# ---------------------------------------------------------------------------
# PHASE-PLOT GRID PARAMETERS
# ---------------------------------------------------------------------------
V_MIN = -85.0
V_MAX = 40.0
U_MIN = -30.0
U_MAX = 160.0
N_GRID = 45

# ---------------------------------------------------------------------------
# PLOT SELECTION / OUTPUT PARAMETERS
# ---------------------------------------------------------------------------
PRE_SPIKE_OFFSET_MS = DT
POST_SPIKE_OFFSET_MS = DT
IMMEDIATE_AUTAPSE_OFFSET_MS = DT
LATE_AUTAPSE_OFFSET_MS = 10.0
AUTAPSE_DIED_DOWN_FRACTION = 0.01
TRAJECTORY_WINDOW_MS = 35.0
PHASE_OUTPUT_PATH = Path("aqua_RS_spike_autapse_vs_control_phase_plot.png")
AUTAPTIC_TIME_SERIES_OUTPUT_PATH = Path("aqua_RS_spike_autapse_time_series.png")
NON_AUTAPTIC_TIME_SERIES_OUTPUT_PATH = Path("aqua_RS_spike_no_autapse_time_series.png")
DPI = 300


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "AQUA_package") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "AQUA_package"))


def import_aqua():
    """Import AQUA, tolerating a missing brian2 install when it is unused here."""
    try:
        from aqua.AQUA_general import AQUA

        return AQUA
    except ModuleNotFoundError as exc:
        if exc.name != "brian2":
            raise

    sys.modules.setdefault("brian2", types.ModuleType("brian2"))
    from aqua.AQUA_general import AQUA

    return AQUA


AQUA = import_aqua()
from aqua.plotting_functions import plot_membrane_variables


def calculate_phase_field(neuron, v_grid, u_grid, w_delivered, i_inj):
    """
    Calculate dv/dt and du/dt across the v-u grid using AQUA.neuron_model.

    The 2D phase portrait treats the delivered autapse current as frozen at a
    selected time point. This shows the instantaneous v-u vector field during
    autapse-current initiation.
    """
    w_state = np.full_like(v_grid, w_delivered, dtype=float)
    state = np.array([v_grid, u_grid, w_state])
    derivatives = neuron.neuron_model(state, w_delivered, i_inj)
    return derivatives[0], derivatives[1]


def simulate_neuron(params):
    n_iter = int(T_MAX / DT)
    i_trace = I_INJ * np.ones(n_iter)

    neuron = AQUA(params)
    neuron.Initialise(X_START, T_START)
    x_trace, t_trace, spikes = neuron.update_RK2(DT, n_iter, i_trace)

    if len(spikes) == 0:
        raise RuntimeError(
            "No spikes were produced. Increase I_INJ or T_MAX in the parameter block."
        )

    return neuron, x_trace, t_trace, spikes


def index_at_time(t_trace, target_time):
    return int(np.clip(np.searchsorted(t_trace, target_time), 0, len(t_trace) - 1))


def choose_snapshot_indices(t_trace, x_trace, spikes):
    """
    Choose five event-relative snapshots around the first autaptic-neuron spike:
    just before the spike, after the spike but before delayed feedback,
    immediately after delayed feedback, later during autapse decay, and either
    when the delivered autapse current has almost fully decayed or the latest
    possible time before the second spike/reset.
    """
    first_spike_time = spikes[0]
    if len(spikes) >= 2:
        latest_snapshot_time = spikes[1] - DT
    else:
        latest_snapshot_time = t_trace[-1]

    autapse_start_time = first_spike_time + AUTAPTIC_NEURON_PARAMS["tau"]

    if AUTAPTIC_NEURON_PARAMS["tau"] <= DT:
        raise RuntimeError(
            "The configured tau is too short to show a post-spike/pre-autapse panel. "
            "Increase AUTAPTIC_NEURON_PARAMS['tau'] or reduce DT."
        )

    post_spike_offset = min(
        POST_SPIKE_OFFSET_MS,
        max(DT, 0.5 * AUTAPTIC_NEURON_PARAMS["tau"]),
    )
    autapse_start_idx = index_at_time(t_trace, autapse_start_time)
    w_trace = x_trace[2]
    latest_snapshot_idx = index_at_time(t_trace, latest_snapshot_time)
    active_w = w_trace[autapse_start_idx : latest_snapshot_idx + 1]
    peak_w = np.max(active_w)

    if peak_w <= 0.0:
        raise RuntimeError(
            "No delivered autapse current was found. Check e, f, tau, DT, and T_MAX."
        )

    died_down_threshold = AUTAPSE_DIED_DOWN_FRACTION * peak_w
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
            first_spike_time - PRE_SPIKE_OFFSET_MS,
            first_spike_time + post_spike_offset,
            autapse_start_time + IMMEDIATE_AUTAPSE_OFFSET_MS,
            autapse_start_time + LATE_AUTAPSE_OFFSET_MS,
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


def plot_phase_comparison(
    autaptic_neuron,
    autaptic_trace,
    autaptic_spikes,
    non_autaptic_neuron,
    non_autaptic_trace,
    non_autaptic_spikes,
    t_trace,
    snapshot_indices,
    snapshot_labels,
):
    v = np.linspace(V_MIN, V_MAX, N_GRID)
    u = np.linspace(U_MIN, U_MAX, N_GRID)
    v_grid, u_grid = np.meshgrid(v, u)

    fig, axes = plt.subplots(
        2,
        len(snapshot_indices),
        figsize=(4.6 * len(snapshot_indices), 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    pad = int(8.0 / DT)
    traj_start = max(snapshot_indices[0] - pad, 0)
    default_traj_stop = min(
        snapshot_indices[-1] + int(TRAJECTORY_WINDOW_MS / DT),
        len(t_trace) - 1,
    )

    plot_specs = [
        ("autaptic", autaptic_neuron, autaptic_trace, autaptic_spikes, axes[0]),
        (
            "non-autaptic",
            non_autaptic_neuron,
            non_autaptic_trace,
            non_autaptic_spikes,
            axes[1],
        ),
    ]

    for row_label, neuron, x_trace, spikes, row_axes in plot_specs:
        if len(spikes) >= 2:
            traj_stop = index_at_time(t_trace, spikes[1]) + 1
        else:
            traj_stop = default_traj_stop

        for ax, idx, label in zip(row_axes, snapshot_indices, snapshot_labels):
            w_delivered = x_trace[2, idx]
            dv, du = calculate_phase_field(neuron, v_grid, u_grid, w_delivered, I_INJ)
            speed = np.hypot(dv, du)

            ax.streamplot(
                v_grid,
                u_grid,
                dv,
                du,
                color=speed,
                cmap="viridis",
                density=1.15,
                linewidth=0.9,
                arrowsize=1.0,
            )
            ax.plot(
                x_trace[0, traj_start:traj_stop],
                x_trace[1, traj_start:traj_stop],
                color="black",
                lw=1.6,
                alpha=0.75,
                label="trajectory",
            )
            ax.scatter(
                x_trace[0, idx],
                x_trace[1, idx],
                color="crimson",
                s=35,
                zorder=4,
                label="snapshot",
            )
            ax.axvline(AUTAPTIC_NEURON_PARAMS["v_peak"], color="0.35", lw=0.9, ls=":")
            ax.set_title(
                f"{label}\n{row_label}: t = {t_trace[idx]:.2f} ms, "
                f"w_delay = {w_delivered:.2f} pA"
            )
            ax.set_xlabel("v (mV)")
            ax.set_xlim(V_MIN, V_MAX)
            ax.set_ylim(U_MIN, U_MAX)

    axes[0, 0].set_ylabel("autaptic\nu")
    axes[1, 0].set_ylabel("non-autaptic\nu")
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        "AQUA RS phase fields during spike and autapse-current decay",
        fontsize=13,
    )
    return fig


def plot_phase_snapshots(neuron, x_trace, t_trace, snapshot_indices, snapshot_labels):
    v = np.linspace(V_MIN, V_MAX, N_GRID)
    u = np.linspace(U_MIN, U_MAX, N_GRID)
    v_grid, u_grid = np.meshgrid(v, u)

    fig, axes = plt.subplots(
        1,
        len(snapshot_indices),
        figsize=(4.6 * len(snapshot_indices), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(snapshot_indices) == 1:
        axes = [axes]

    first_idx = snapshot_indices[0]
    last_idx = snapshot_indices[-1]
    pad = int(8.0 / DT)
    traj_start = max(first_idx - pad, 0)
    traj_stop = min(last_idx + int(TRAJECTORY_WINDOW_MS / DT), len(t_trace) - 1)

    for ax, idx, label in zip(axes, snapshot_indices, snapshot_labels):
        w_delivered = x_trace[2, idx]
        dv, du = calculate_phase_field(neuron, v_grid, u_grid, w_delivered, I_INJ)
        speed = np.hypot(dv, du)

        ax.streamplot(
            v_grid,
            u_grid,
            dv,
            du,
            color=speed,
            cmap="viridis",
            density=1.15,
            linewidth=0.9,
            arrowsize=1.0,
        )
        ax.plot(
            x_trace[0, traj_start:traj_stop],
            x_trace[1, traj_start:traj_stop],
            color="black",
            lw=1.6,
            alpha=0.75,
            label="trajectory",
        )
        ax.scatter(
            x_trace[0, idx],
            x_trace[1, idx],
            color="crimson",
            s=35,
            zorder=4,
            label="snapshot",
        )
        ax.axvline(AUTAPTIC_NEURON_PARAMS["v_peak"], color="0.35", lw=0.9, ls=":")
        ax.set_title(f"{label}\nt = {t_trace[idx]:.2f} ms, w_delay = {w_delivered:.2f} pA")
        ax.set_xlabel("v (mV)")
        ax.set_xlim(V_MIN, V_MAX)
        ax.set_ylim(U_MIN, U_MAX)

    axes[0].set_ylabel("u")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        "AQUA RS model phase field during spike-triggered autapse-current initiation",
        fontsize=13,
    )
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


def main():
    autaptic_neuron, autaptic_trace, t_trace, autaptic_spikes = simulate_neuron(
        AUTAPTIC_NEURON_PARAMS
    )
    non_autaptic_neuron, non_autaptic_trace, _, non_autaptic_spikes = simulate_neuron(
        NON_AUTAPTIC_NEURON_PARAMS
    )
    snapshot_indices, snapshot_labels = choose_snapshot_indices(
        t_trace,
        autaptic_trace,
        autaptic_spikes,
    )

    autaptic_time_series_fig = plot_time_series(
        autaptic_trace,
        t_trace,
        autaptic_spikes,
        snapshot_indices,
        snapshot_labels,
        "AQUA RS autaptic neuron variables during spike and autapse decay",
    )
    autaptic_time_series_fig.savefig(AUTAPTIC_TIME_SERIES_OUTPUT_PATH, dpi=DPI)

    non_autaptic_time_series_fig = plot_time_series(
        non_autaptic_trace,
        t_trace,
        non_autaptic_spikes,
        snapshot_indices,
        snapshot_labels,
        "AQUA RS non-autaptic neuron variables at matched autaptic event times",
    )
    non_autaptic_time_series_fig.savefig(NON_AUTAPTIC_TIME_SERIES_OUTPUT_PATH, dpi=DPI)

    phase_fig = plot_phase_comparison(
        autaptic_neuron,
        autaptic_trace,
        autaptic_spikes,
        non_autaptic_neuron,
        non_autaptic_trace,
        non_autaptic_spikes,
        t_trace,
        snapshot_indices,
        snapshot_labels,
    )
    phase_fig.savefig(PHASE_OUTPUT_PATH, dpi=DPI)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    print(f"Saved phase plot to {PHASE_OUTPUT_PATH.resolve()}")
    print(f"Saved autaptic time-series plot to {AUTAPTIC_TIME_SERIES_OUTPUT_PATH.resolve()}")
    print(
        "Saved non-autaptic time-series plot to "
        f"{NON_AUTAPTIC_TIME_SERIES_OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
