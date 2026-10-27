"""
Video of the v-u phase portrait and flow field during a spike and the following
autapse current, autaptic (left) vs non-autaptic (right) neuron.

Reuses the simulation, snapshot selection and flow-field functions from
aqua_spike_autapse_phase_plot.py. The window starts before the first spike and
runs until just before the autaptic neuron's second spike. Nullclines are drawn
in orange (v solid, u dashed) and move with the delivered autapse current.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

import aqua_spike_autapse_phase_plot as app

# ---------------------------------------------------------------------------
# VIDEO PARAMETERS
# ---------------------------------------------------------------------------
FRAME_STRIDE = 2  # simulation steps per video frame
FPS = 20
VIDEO_DPI = 150
VIDEO_OUTPUT_PATH = "aqua_RS_spike_autapse_vs_control_phase_video.mp4"

# Window: from PRE_SPIKE_WINDOW_MS before the first spike until
# END_BEFORE_SECOND_SPIKE_MS before the autaptic neuron's second spike.
PRE_SPIKE_WINDOW_MS = 15.0
END_BEFORE_SECOND_SPIKE_MS = 0.5

# Nullclines: muted orange, distinct from viridis / black / grey / crimson
NULLCLINE_COLOR = "#d9822b"
NULLCLINE_GRID = 300  # finer grid so the contours are smooth


def draw_nullclines(ax, neuron, w_delivered, v_fine, u_fine):
    """Draw the v-nullcline (dv/dt = 0, solid) and u-nullcline (du/dt = 0, dashed)."""
    dv, du = app.calculate_phase_field(neuron, v_fine, u_fine, w_delivered, app.I_INJ)
    style = dict(colors=NULLCLINE_COLOR, linewidths=1.4, alpha=0.85, zorder=3)
    ax.contour(v_fine, u_fine, dv, levels=[0], linestyles="solid", **style)
    ax.contour(v_fine, u_fine, du, levels=[0], linestyles="dashed", **style)


def draw_phase_frame(ax, neuron, x_trace, t_trace, idx, start_idx, context_stop,
                     v_grid, u_grid, v_fine, u_fine, row_label, caption):
    """Redraw one phase-portrait panel for simulation index `idx`."""
    ax.clear()
    w_delivered = x_trace[2, idx]
    dv, du = app.calculate_phase_field(neuron, v_grid, u_grid, w_delivered, app.I_INJ)
    ax.streamplot(
        v_grid, u_grid, dv, du,
        color=np.hypot(dv, du), cmap="viridis",
        density=1.15, linewidth=0.9, arrowsize=1.0,
    )
    draw_nullclines(ax, neuron, w_delivered, v_fine, u_fine)
    ax.plot([], [], color=NULLCLINE_COLOR, lw=1.4, label="v-nullcline")
    ax.plot([], [], color=NULLCLINE_COLOR, lw=1.4, ls="--", label="u-nullcline")
    ax.plot(x_trace[0, :context_stop], x_trace[1, :context_stop],
            color="0.6", lw=1.0, alpha=0.6, label="full trajectory")
    ax.plot(x_trace[0, start_idx:idx + 1], x_trace[1, start_idx:idx + 1],
            color="black", lw=1.8, alpha=0.85, label="trajectory so far")
    ax.scatter(x_trace[0, idx], x_trace[1, idx], color="crimson", s=45, zorder=4,
               label="current state")
    ax.axvline(app.AUTAPTIC_NEURON_PARAMS["v_peak"], color="0.35", lw=0.9, ls=":")
    ax.set_xlim(app.V_MIN, app.V_MAX)
    ax.set_ylim(app.U_MIN, app.U_MAX)
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u")
    ax.set_title(
        f"{row_label}: t = {t_trace[idx]:.2f} ms, w_delay = {w_delivered:.2f} pA\n{caption}"
    )


def main():
    auto_neuron, auto_trace, t_trace, auto_spikes = app.simulate_neuron(
        app.AUTAPTIC_NEURON_PARAMS
    )
    non_neuron, non_trace, _, non_spikes = app.simulate_neuron(
        app.NON_AUTAPTIC_NEURON_PARAMS
    )
    snapshot_indices, snapshot_labels = app.choose_snapshot_indices(
        t_trace, auto_trace, auto_spikes
    )

    start_idx = app.index_at_time(t_trace, auto_spikes[0] - PRE_SPIKE_WINDOW_MS)
    if len(auto_spikes) >= 2:
        stop_idx = app.index_at_time(t_trace, auto_spikes[1] - END_BEFORE_SECOND_SPIKE_MS)
    else:
        stop_idx = len(t_trace) - 1
    frame_indices = np.arange(start_idx, stop_idx + 1, FRAME_STRIDE)
    if frame_indices[-1] != stop_idx:
        frame_indices = np.append(frame_indices, stop_idx)

    # Faint context trajectory: start of sim -> second spike, as in the original figure
    def context_stop(spikes):
        if len(spikes) >= 2:
            return app.index_at_time(t_trace, spikes[1])
        return min(stop_idx + int(app.TRAJECTORY_WINDOW_MS / app.DT), len(t_trace) - 1)

    auto_ctx, non_ctx = context_stop(auto_spikes), context_stop(non_spikes)

    v = np.linspace(app.V_MIN, app.V_MAX, app.N_GRID)
    u = np.linspace(app.U_MIN, app.U_MAX, app.N_GRID)
    v_grid, u_grid = np.meshgrid(v, u)
    v_fine, u_fine = np.meshgrid(
        np.linspace(app.V_MIN, app.V_MAX, NULLCLINE_GRID),
        np.linspace(app.U_MIN, app.U_MAX, NULLCLINE_GRID),
    )

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_auto = fig.add_subplot(gs[0, 0])
    ax_non = fig.add_subplot(gs[0, 1], sharex=ax_auto, sharey=ax_auto)
    ax_v = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1], sharex=ax_v)

    # Time-series panels (static, with a moving cursor)
    window = slice(start_idx, stop_idx + 1)
    t_win = t_trace[window]
    ax_v.plot(t_win, auto_trace[0, window], color="tab:blue", lw=1.4, label="autaptic")
    ax_v.plot(t_win, non_trace[0, window], color="0.45", lw=1.4, ls="--",
              label="non-autaptic")
    ax_v.set_ylabel("v (mV)")
    ax_v.set_xlabel("t (ms)")
    ax_v.legend(loc="upper right", frameon=False, fontsize=8)
    ax_w.plot(t_win, auto_trace[2, window], color="tab:blue", lw=1.4, label="autaptic")
    ax_w.plot(t_win, non_trace[2, window], color="0.45", lw=1.4, ls="--",
              label="non-autaptic")
    ax_w.set_ylabel("w_delay (pA)")
    ax_w.set_xlabel("t (ms)")
    for ax in (ax_v, ax_w):
        for idx in snapshot_indices:
            ax.axvline(t_trace[idx], color="crimson", lw=0.6, alpha=0.4)
        ax.set_xlim(t_win[0], t_win[-1])
    cursors = [ax.axvline(t_trace[start_idx], color="black", lw=1.2) for ax in (ax_v, ax_w)]

    fig.suptitle(
        "AQUA RS phase field during spike and autapse current: autaptic vs non-autaptic",
        fontsize=13,
    )

    def caption_for(idx):
        passed = [lab for i, lab in zip(snapshot_indices, snapshot_labels) if idx >= i]
        return passed[-1] if passed else "before spike"

    def update(frame_number):
        idx = frame_indices[frame_number]
        caption = caption_for(idx)
        draw_phase_frame(ax_auto, auto_neuron, auto_trace, t_trace, idx, start_idx,
                         auto_ctx, v_grid, u_grid, v_fine, u_fine, "autaptic", caption)
        draw_phase_frame(ax_non, non_neuron, non_trace, t_trace, idx, start_idx,
                         non_ctx, v_grid, u_grid, v_fine, u_fine, "non-autaptic", caption)
        ax_auto.legend(loc="upper left", frameon=False, fontsize=8)
        for cursor in cursors:
            cursor.set_xdata([t_trace[idx], t_trace[idx]])
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=4000)

    def progress(i, n):
        if i % 25 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}")

    print(
        f"Rendering {len(frame_indices)} frames "
        f"(t = {t_trace[start_idx]:.2f} -> {t_trace[stop_idx]:.2f} ms)"
    )
    anim.save(VIDEO_OUTPUT_PATH, writer=writer, dpi=VIDEO_DPI, progress_callback=progress)
    print(f"Saved video to {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
