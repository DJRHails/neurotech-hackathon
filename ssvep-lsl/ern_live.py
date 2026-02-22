"""Live ERN detection and visualisation from an LSL stream.

Connects to an LSL EEG stream (default: ern_eeg), applies a 1-30 Hz
bandpass filter, and displays a dark-themed two-panel matplotlib plot:
  - Top: All channels stacked as an EEG strip chart with ERN markers
  - Bottom: Running negative-peak envelope with detection threshold

Detected ERN events are printed to the console.

Usage:
    uv run python ern_live.py
    uv run python ern_live.py -s ern_eeg --threshold -5.0
"""

import argparse
from collections import deque

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfilt, sosfilt_zi

from ern_detector import ERNDetector

PLOT_SECONDS = 5
FCZ_INDEX = 0
MIN_PLOT_SAMPLES = 10
ENV_WINDOW_SEC = 0.15
CHANNEL_LABELS = ["FCz", "Fz", "Cz"]

# -- Theme ---------------------------------------------------------
BG_COLOR = "#0c0c14"
PANEL_BG = "#10101c"
GRID_COLOR = "#1a1a2e"
TEXT_COLOR = "#c0c0d0"
ACCENT_DIM = "#2a2a40"

CHAN_COLORS = ["#00e5ff", "#69f0ae", "#ffab40"]
CHAN_GLOW = ["#00e5ff30", "#69f0ae30", "#ffab4030"]

ERN_MARKER_COLOR = "#ff1744"
ERN_SPAN_COLOR = "#ff174418"
THRESHOLD_COLOR = "#ff4081"
ENV_LINE_COLOR = "#b388ff"
ENV_GLOW_COLOR = "#b388ff25"


def _build_ern_filter(
    fs: float,
    n_ch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tighter bandpass for ERN detection (1-30 Hz).

    Args:
        fs: Sampling frequency in Hz.
        n_ch: Number of EEG channels.
    """
    bp = butter(2, [1.0, 30.0], btype="band", fs=fs, output="sos")
    n50 = butter(2, [49.0, 51.0], btype="bandstop", fs=fs, output="sos")
    n60 = butter(2, [59.0, 61.0], btype="bandstop", fs=fs, output="sos")
    sos = np.vstack([bp, n50, n60])
    zi_single = sosfilt_zi(sos)
    zi = np.repeat(zi_single[:, :, np.newaxis], n_ch, axis=2)
    return sos, zi


def _style_axis(ax: plt.Axes) -> None:
    """Apply dark theme to a single axis."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(ACCENT_DIM)
        spine.set_linewidth(0.5)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.6)


def _init_plot(sr: int, buf_len: int, threshold: float, n_ch: int) -> dict:
    """Set up dark-themed two-panel figure for ERN monitoring."""
    plt.ion()
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.size": 9,
        }
    )

    fig, (ax_channels, ax_envelope) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.28},
        num="ERN Live Monitor",
    )
    fig.patch.set_facecolor(BG_COLOR)
    _style_axis(ax_channels)
    _style_axis(ax_envelope)

    # -- Top panel: stacked channel traces --------------------------
    ch_lines = []
    ch_glows = []
    for i in range(min(n_ch, len(CHANNEL_LABELS))):
        (glow,) = ax_channels.plot(
            [],
            [],
            linewidth=3.5,
            color=CHAN_GLOW[i],
            solid_capstyle="round",
        )
        (line,) = ax_channels.plot(
            [],
            [],
            linewidth=0.9,
            color=CHAN_COLORS[i],
            label=CHANNEL_LABELS[i],
            solid_capstyle="round",
        )
        ch_glows.append(glow)
        ch_lines.append(line)

    ax_channels.set_ylabel("Channels (uV + offset)", fontsize=9)
    ax_channels.set_xlabel("Time (s)", fontsize=9)
    ax_channels.set_xlim(0, buf_len / sr)
    ax_channels.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.3,
        edgecolor=ACCENT_DIM,
        facecolor=PANEL_BG,
        labelcolor=TEXT_COLOR,
    )

    # -- Bottom panel: envelope with threshold ----------------------
    (env_glow,) = ax_envelope.plot(
        [],
        [],
        linewidth=4.0,
        color=ENV_GLOW_COLOR,
        solid_capstyle="round",
    )
    (env_line,) = ax_envelope.plot(
        [],
        [],
        linewidth=1.2,
        color=ENV_LINE_COLOR,
        label="Neg. peak (150ms)",
        solid_capstyle="round",
    )
    ax_envelope.axhline(
        y=threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=f"Threshold ({threshold} uV)",
    )
    ax_envelope.set_ylabel("Min Amp (uV)", fontsize=9)
    ax_envelope.set_xlabel("Time (s)", fontsize=9)
    ax_envelope.set_xlim(0, buf_len / sr)
    ax_envelope.set_ylim(-30, 10)
    ax_envelope.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.3,
        edgecolor=ACCENT_DIM,
        facecolor=PANEL_BG,
        labelcolor=TEXT_COLOR,
    )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.07)
    fig.canvas.draw()
    fig.show()

    return {
        "fig": fig,
        "ax_channels": ax_channels,
        "ax_envelope": ax_envelope,
        "ch_lines": ch_lines,
        "ch_glows": ch_glows,
        "env_line": env_line,
        "env_glow": env_glow,
        "ern_markers": [],
    }


def _update_ern_markers(
    plot: dict,
    ern_hit_samples: list[int],
    count: int,
    ring_len: int,
    sr: int,
) -> None:
    """Redraw ERN markers on the channel panel and prune old hits."""
    for artist in plot["ern_markers"]:
        artist.remove()
    plot["ern_markers"] = []

    ax = plot["ax_channels"]
    buf_start = count - ring_len
    for abs_s in ern_hit_samples:
        rel = abs_s - buf_start
        if 0 <= rel < ring_len:
            x = rel / sr
            span = ax.axvspan(
                x - 0.05,
                x + 0.05,
                color=ERN_SPAN_COLOR,
                zorder=0,
            )
            line = ax.axvline(
                x=x,
                color=ERN_MARKER_COLOR,
                linewidth=1.0,
                alpha=0.8,
                zorder=5,
            )
            plot["ern_markers"].extend([span, line])

    ern_hit_samples[:] = [s for s in ern_hit_samples if s >= buf_start]


def _update_plot(
    plot: dict,
    ring: deque,
    env_ring: deque,
    ern_hit_samples: list[int],
    count: int,
    sr: int,
    n_ch: int,
    stream_name: str,
    use_filter: bool,
    total_hits: int,
) -> None:
    """Refresh both panels with current buffer data."""
    if len(ring) <= MIN_PLOT_SAMPLES:
        return

    data = np.array(ring)
    t = np.arange(len(data)) / sr
    n_disp = min(n_ch, len(CHANNEL_LABELS))

    # Stack channels with vertical offsets
    spacing = 60.0
    for i in range(n_disp):
        offset = (n_disp - 1 - i) * spacing
        y = data[:, i] + offset
        plot["ch_lines"][i].set_data(t, y)
        plot["ch_glows"][i].set_data(t, y)

    ax_ch = plot["ax_channels"]
    ax_ch.set_xlim(0, t[-1])
    y_lo = -spacing * 0.4
    y_hi = (n_disp - 1) * spacing + spacing * 0.4
    ax_ch.set_ylim(y_lo, y_hi)

    # Channel name ticks on the y-axis
    tick_pos = [(n_disp - 1 - i) * spacing for i in range(n_disp)]
    ax_ch.set_yticks(tick_pos)
    ax_ch.set_yticklabels(CHANNEL_LABELS[:n_disp])
    for label, color in zip(ax_ch.get_yticklabels(), CHAN_COLORS, strict=False):
        label.set_color(color)

    _update_ern_markers(plot, ern_hit_samples, count, len(ring), sr)

    # Bottom panel: envelope
    if len(env_ring) > MIN_PLOT_SAMPLES:
        env_data = np.array(env_ring)
        t_env = np.arange(len(env_data)) / sr
        plot["env_line"].set_data(t_env, env_data)
        plot["env_glow"].set_data(t_env, env_data)
        plot["ax_envelope"].set_xlim(0, t_env[-1])
        plot["ax_envelope"].relim()
        plot["ax_envelope"].autoscale_view(scalex=False, scaley=True)

    filt_str = "BP 1-30 Hz" if use_filter else "Raw"
    hit_label = f"Hits: {total_hits}" if total_hits else "No hits"
    plot["fig"].suptitle(
        f"ERN Monitor  //  {stream_name}  //  "
        f"{n_ch}ch @ {sr}Hz  //  "
        f"{count:,} samples  //  "
        f"{filt_str}  //  {hit_label}",
        fontsize=10,
        color=TEXT_COLOR,
        fontfamily="monospace",
    )


def _connect_lsl(stream_name: str) -> tuple[StreamInlet, int, int]:
    """Resolve and connect to an LSL stream."""
    logger.info("Resolving LSL stream '{}'...", stream_name)
    streams = resolve_byprop("name", stream_name, minimum=1, timeout=10)
    if not streams:
        logger.error("No stream '{}' found.", stream_name)
        raise SystemExit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    n_ch = info.channel_count()
    sr = int(info.nominal_srate())
    logger.info("Connected: {}, {} ch @ {} Hz", info.name(), n_ch, sr)
    return inlet, n_ch, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ERN detection from LSL stream")
    parser.add_argument(
        "-s",
        "--stream",
        default="ern_eeg",
        help="LSL stream name (default: ern_eeg)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=-5.0,
        help="ERN detection threshold in uV (default: -5.0)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip bandpass filtering",
    )
    args = parser.parse_args()

    inlet, n_ch, sr = _connect_lsl(args.stream)

    use_filter = not args.no_filter
    sos, zi = None, None
    if use_filter:
        sos, zi = _build_ern_filter(float(sr), n_ch)
        logger.info("Filter: bandpass 1-30 Hz, notch 50+60 Hz")

    detector = ERNDetector(
        s_rate=sr,
        frontal_channels=list(range(min(n_ch, 3))),
        threshold_uv=args.threshold,
    )

    buf_len = PLOT_SECONDS * sr
    ring: deque[list[float]] = deque(maxlen=buf_len)
    env_ring: deque[float] = deque(maxlen=buf_len)
    ern_hit_samples: list[int] = []
    env_window = int(ENV_WINDOW_SEC * sr)
    count = 0

    plot = _init_plot(sr, buf_len, args.threshold, n_ch)
    logger.info("Monitoring for ERN events (Ctrl+C to stop)")

    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=64)
            if not chunk:
                plt.pause(0.01)
                continue

            samples = np.array(chunk, dtype=np.float32)
            if use_filter:
                samples, zi = sosfilt(sos, samples, axis=0, zi=zi)
                samples = samples.astype(np.float32)

            if timestamps:
                hits = detector.detect(samples, timestamps[-1])
                for hit in hits:
                    ern_hit_samples.append(count + len(samples))
                    logger.info(
                        "[ERN] Hit #{} | ch={} | amp={:.1f} uV",
                        detector.total_hits,
                        hit.channel,
                        hit.amplitude,
                    )

            for row in samples:
                ring.append(row.tolist())
                count += 1
                if len(ring) >= env_window:
                    recent = [ring[-j - 1][FCZ_INDEX] for j in range(env_window)]
                    env_ring.append(float(min(recent)))
                else:
                    env_ring.append(float(row[FCZ_INDEX]))

            _update_plot(
                plot,
                ring,
                env_ring,
                ern_hit_samples,
                count,
                sr,
                n_ch,
                args.stream,
                use_filter,
                detector.total_hits,
            )
            plt.pause(0.001)

    except KeyboardInterrupt:
        logger.info(
            "Stopped. {} samples, {} ERN hits.",
            count,
            detector.total_hits,
        )
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
