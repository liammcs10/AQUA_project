"""
- - - main.py - - -

Generates AQUA RS phase plots (flow field + nullclines + trajectory) and
matching time-series plots for an autaptic neuron and its non-autaptic
control, around the neuron's first spike.

All simulation, model, grid, snapshot, and figure parameters are read from
a config file (see RS_config.ini) so the study can be reproduced or swept
without editing this file.

Usage
-----
    python main.py --config RS_config.ini [--save]
"""

import numpy as np
import matplotlib.pyplot as plt

import CLI
import CF
from functions import (
    cast_to_float,
    parse_bool,
    no_autapse_variant,
    simulate_trajectory,
    choose_snapshot_indices,
    compute_trajectory_window,
    build_grid_batch,
    plot_phase_grid,
    plot_time_series,
)


def build_row_spec(
    label, params, x_trace, t_trace, spikes, v_grid, snapshot_indices,
    trajectory_window_ms, trajectory_cutoff_ms, dt,
):
    """Bundle everything plot_phase_grid needs for one row (autaptic/control)."""
    traj_start, traj_stop = compute_trajectory_window(
        t_trace, spikes, snapshot_indices, dt, trajectory_window_ms, trajectory_cutoff_ms
    )

    return {
        "label": label,
        "params": params,
        "grid_batch": build_grid_batch(params, v_grid),
        "x_trace": x_trace,
        "t_trace": t_trace,
        "traj_start": traj_start,
        "traj_stop": traj_stop,
    }


def main():
    args = CLI.command_line()

    if args.config is None:
        print("No config file passed... exiting")
        quit()

    conf = CF.read_conf(args.config)
    print("Config extracted...")

    neuron_params = cast_to_float(conf["Neuron"])
    non_autaptic_params = no_autapse_variant(neuron_params)

    sim = cast_to_float(conf["Simulation"])
    dt = sim["dt"]
    T = sim["T"]
    I_inj = sim["I_inj"]
    x_start = [sim["v_start"], sim["u_start"], sim["w_start"]]
    t_start = sim.get("t_start", 0.0)

    grid = cast_to_float(conf["Grid"])
    v_min, v_max = grid["v_min"], grid["v_max"]
    u_min, u_max = grid["u_min"], grid["u_max"]
    n_grid = int(grid["n_grid"])

    snap = cast_to_float(conf["Snapshots"])
    trajectory_cutoff_ms = snap.get("trajectory_cutoff_ms")
    if not isinstance(trajectory_cutoff_ms, float):
        trajectory_cutoff_ms = None

    fig_conf = conf["Figure"]
    show_nullclines = parse_bool(fig_conf.get("show_nullclines", "true"))
    density = float(fig_conf.get("density", 1.15))
    linewidth = float(fig_conf.get("linewidth", 0.9))
    dpi = int(float(fig_conf.get("dpi", 300)))

    output = conf["Output"]

    print(f"Simulating '{neuron_params['name']}' (autaptic) and its non-autaptic control...")
    autaptic_trace, t_trace, autaptic_spikes = simulate_trajectory(
        neuron_params, T, dt, I_inj, x_start, t_start
    )
    non_autaptic_trace, _, non_autaptic_spikes = simulate_trajectory(
        non_autaptic_params, T, dt, I_inj, x_start, t_start
    )

    snapshot_indices, snapshot_labels = choose_snapshot_indices(
        t_trace,
        autaptic_trace,
        autaptic_spikes,
        tau=neuron_params["tau"],
        dt=dt,
        pre_spike_offset_ms=snap["pre_spike_offset_ms"],
        post_spike_offset_ms=snap["post_spike_offset_ms"],
        immediate_autapse_offset_ms=snap["immediate_autapse_offset_ms"],
        late_autapse_offset_ms=snap["late_autapse_offset_ms"],
        autapse_died_down_fraction=snap["autapse_died_down_fraction"],
    )

    v_lin = np.linspace(v_min, v_max, n_grid)
    u_lin = np.linspace(u_min, u_max, n_grid)
    v_grid, u_grid = np.meshgrid(v_lin, u_lin)

    autaptic_time_series_fig = plot_time_series(
        autaptic_trace,
        t_trace,
        autaptic_spikes,
        snapshot_indices,
        snapshot_labels,
        f"AQUA {neuron_params['name']} autaptic neuron variables during spike and autapse decay",
    )
    non_autaptic_time_series_fig = plot_time_series(
        non_autaptic_trace,
        t_trace,
        non_autaptic_spikes,
        snapshot_indices,
        snapshot_labels,
        f"AQUA {neuron_params['name']} non-autaptic neuron variables at matched event times",
    )

    row_specs = [
        build_row_spec(
            "autaptic", neuron_params, autaptic_trace, t_trace, autaptic_spikes,
            v_grid, snapshot_indices, snap["trajectory_window_ms"], trajectory_cutoff_ms, dt,
        ),
        build_row_spec(
            "non-autaptic", non_autaptic_params, non_autaptic_trace, t_trace, non_autaptic_spikes,
            v_grid, snapshot_indices, snap["trajectory_window_ms"], trajectory_cutoff_ms, dt,
        ),
    ]

    phase_fig = plot_phase_grid(
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
        f"AQUA {neuron_params['name']} phase fields during spike and autapse-current decay",
    )

    if args.save:
        autaptic_time_series_fig.savefig(output["autaptic_time_series"], dpi=dpi)
        non_autaptic_time_series_fig.savefig(output["non_autaptic_time_series"], dpi=dpi)
        phase_fig.savefig(output["phase_plot"], dpi=dpi)
        print(f"Saved autaptic time-series plot to {output['autaptic_time_series']}")
        print(f"Saved non-autaptic time-series plot to {output['non_autaptic_time_series']}")
        print(f"Saved phase plot to {output['phase_plot']}")

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
