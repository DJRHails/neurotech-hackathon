"""Live peak-to-peak amplitude plot with ERN event markers.

Reads EEG via LSL, applies bandpass+notch filtering, computes
sliding-window P2P amplitude on frontal channels, runs ERN
detection, and marks detected ERN events on the plot.

Usage:
    uv run python plot_p2p.py -s ern_eeg
    uv run python plot_p2p.py -s ern_eeg -w 0.5 --history 60
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from matplotlib import animation  # noqa: E402
from pylsl import StreamInlet, resolve_byprop  # noqa: E402

from ern_detector import ERNDetector  # noqa: E402
from filters import apply_filter, build_filter  # noqa: E402

DEFAULT_WINDOW_SEC = 1.0
DEFAULT_HISTORY_SEC = 60


MARKER_STREAM_NAME = "ern_markers"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live P2P amplitude plot with ERN markers",
    )
    parser.add_argument(
        "-s",
        "--stream",
        default="ern_eeg",
        help="LSL stream name (default: ern_eeg)",
    )
    parser.add_argument(
        "-m",
        "--markers",
        default=MARKER_STREAM_NAME,
        help=f"LSL marker stream name (default: {MARKER_STREAM_NAME})",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="P2P window size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--history",
        type=float,
        default=DEFAULT_HISTORY_SEC,
        help="Seconds of history to display (default: 30)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=-5.0,
        help="ERN detection threshold in uV (default: -5.0)",
    )
    return parser.parse_args()


def _connect_lsl(stream_name: str) -> tuple[StreamInlet, int, int, list[str]]:
    """Resolve LSL stream, return (inlet, s_rate, n_ch, ch_names)."""
    print(f"Resolving LSL stream '{stream_name}'...")
    streams = resolve_byprop("name", stream_name, minimum=1, timeout=10)
    if not streams:
        msg = f"No LSL stream '{stream_name}' found."
        raise RuntimeError(msg)
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    s_rate = int(info.nominal_srate())
    n_ch = info.channel_count()

    # Extract channel names from LSL metadata
    ch_names: list[str] = []
    ch_xml = info.desc().child("channels")
    if not ch_xml.empty():
        ch_node = ch_xml.child("channel")
        while not ch_node.empty():
            label = ch_node.child_value("label")
            ch_names.append(label or f"ch{len(ch_names)}")
            ch_node = ch_node.next_sibling()
    if len(ch_names) != n_ch:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    print(f"Connected: {info.name()}, {n_ch} ch ({ch_names}) @ {s_rate} Hz")
    return inlet, s_rate, n_ch, ch_names


def _connect_markers(stream_name: str) -> StreamInlet | None:
    """Try to resolve a marker stream. Returns None if not found."""
    print(f"Looking for marker stream '{stream_name}'...")
    streams = resolve_byprop("name", stream_name, minimum=1, timeout=3)
    if not streams:
        print(f"  No marker stream '{stream_name}' found, continuing without.")
        return None
    inlet = StreamInlet(streams[0])
    print(f"  Marker stream connected: {inlet.info().name()}")
    return inlet


@dataclass
class PlotState:
    """Mutable state for the animation update loop."""

    eeg_buf: deque
    p2p_history: dict[int, deque]
    time_history: deque
    ern_times: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    total_windows: int = 0
    total_ern_hits: int = 0


def _setup_plot(
    channels: list[int],
    ch_names: list[str],
    window_sec: float,
) -> tuple[plt.Figure, plt.Axes, dict[int, plt.Line2D], plt.Text]:
    """Create the matplotlib figure and channel lines."""
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(channels), 1)))
    fig, ax = plt.subplots(figsize=(12, 5))

    lines: dict[int, plt.Line2D] = {}
    for i, ch in enumerate(channels):
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.5, label=ch_names[ch])
        lines[ch] = line

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P2P Amplitude (uV)")
    ax.set_title(f"Peak-to-Peak Amplitude ({window_sec}s window) — ERN events marked")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ern_label = ax.text(
        0.01,
        0.95,
        "ERN: 0/0 (0%)",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="red",
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "red", "pad": 4},
    )

    return fig, ax, lines, ern_label


def _prune_vlines(
    vline_refs: list[plt.Line2D],
    visible_start: float,
) -> None:
    """Remove vertical lines that have scrolled off the visible area."""
    for vl in list(vline_refs):
        xdata = vl.get_xdata()
        if len(xdata) > 0 and xdata[0] < visible_start:
            vl.remove()
            vline_refs.remove(vl)


def _rescale_axes(
    ax: plt.Axes,
    t: np.ndarray,
    channels: list[int],
    p2p_history: dict[int, deque],
    *vline_lists: list[plt.Line2D],
) -> None:
    """Rescale axes to fit data and prune off-screen markers."""
    if len(t) > 1:
        ax.set_xlim(t[0], t[-1])
    y_vals = [v for ch in channels for v in p2p_history[ch]]
    if y_vals:
        y_max = max(y_vals) * 1.1 or 1.0
        ax.set_ylim(0, y_max)

    visible_start = t[0] if len(t) > 0 else 0.0
    for refs in vline_lists:
        _prune_vlines(refs, visible_start)


def _pull_markers(
    marker_inlet: StreamInlet | None,
    elapsed: float,
    ax: plt.Axes,
    marker_line_refs: list[plt.Line2D],
) -> None:
    """Pull any pending markers and draw them on the plot."""
    if marker_inlet is None:
        return
    sample, _ = marker_inlet.pull_sample(timeout=0.0)
    while sample is not None:
        desc = sample[0]
        is_error = "ERROR" in desc
        color = "#FF9800" if is_error else "#42A5F5"
        style = "-" if is_error else ":"
        alpha = 0.7 if is_error else 0.4
        vline = ax.axvline(elapsed, color=color, alpha=alpha, linewidth=1.2, linestyle=style)
        marker_line_refs.append(vline)
        label = "ERROR" if is_error else "correct"
        print(f"[MARKER] t={elapsed:.1f}s {label}: {desc}")
        sample, _ = marker_inlet.pull_sample(timeout=0.0)


def _make_updater(
    inlet: StreamInlet,
    marker_inlet: StreamInlet | None,
    s_rate: int,
    channels: list[int],
    window_samples: int,
    state: PlotState,
    ax: plt.Axes,
    lines: dict[int, plt.Line2D],
    ern_label: plt.Text,
    sos: np.ndarray,
    zi: np.ndarray,
    detector: ERNDetector,
) -> callable:
    """Build the FuncAnimation update closure."""
    zi_state = [zi]
    ern_line_refs: list[plt.Line2D] = []
    marker_line_refs: list[plt.Line2D] = []

    def _update(_frame: int) -> list:
        # Always pull markers every frame so none are missed
        _pull_markers(marker_inlet, state.elapsed, ax, marker_line_refs)

        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=64)
        if not chunk:
            return list(lines.values())

        samples = np.array(chunk, dtype=np.float32)
        filtered, zi_state[0] = apply_filter(samples, sos, zi_state[0])

        for row in filtered:
            state.eeg_buf.append(row)

        state.elapsed += len(chunk) / s_rate

        if len(state.eeg_buf) < window_samples:
            return list(lines.values())

        buf_arr = np.array(state.eeg_buf)
        state.time_history.append(state.elapsed)

        state.total_windows += 1
        for ch in channels:
            state.p2p_history[ch].append(float(np.ptp(buf_arr[:, ch])))

        # ERN detection
        hits = detector.detect(filtered, time.time())
        for hit in hits:
            state.total_ern_hits += 1
            state.ern_times.append(state.elapsed)
            vline = ax.axvline(
                state.elapsed,
                color="#4CAF50",
                alpha=0.6,
                linewidth=1.5,
                linestyle="--",
            )
            ern_line_refs.append(vline)
            print(f"[ERN] t={state.elapsed:.1f}s ch={hit.channel} amp={hit.amplitude:.1f} uV")

        # Update running ERN percentage and colour
        pct = state.total_ern_hits / state.total_windows * 100
        ern_label.set_text(f"ERN: {state.total_ern_hits}/{state.total_windows} ({pct:.1f}%)")
        color = "#4CAF50" if hits else "#F44336"
        ern_label.set_color(color)
        ern_label.get_bbox_patch().set_edgecolor(color)

        # Update line data
        t = np.array(state.time_history)
        for ch in channels:
            vals = np.array(state.p2p_history[ch])
            lines[ch].set_data(t[-len(vals) :], vals)

        _rescale_axes(ax, t, channels, state.p2p_history, ern_line_refs, marker_line_refs)
        return list(lines.values())

    return _update


def main() -> None:
    args = _parse_args()
    inlet, s_rate, n_ch, ch_names = _connect_lsl(args.stream)
    marker_inlet = _connect_markers(args.markers)
    channels = list(range(n_ch))

    window_samples = int(args.window * s_rate)
    max_points = int(args.history / args.window)

    sos, zi = build_filter(float(s_rate), n_ch)
    detector = ERNDetector(s_rate=s_rate, threshold_uv=args.threshold)

    state = PlotState(
        eeg_buf=deque(maxlen=window_samples),
        p2p_history={ch: deque(maxlen=max_points) for ch in channels},
        time_history=deque(maxlen=max_points),
    )

    fig, ax, lines, ern_label = _setup_plot(channels, ch_names, args.window)
    update_fn = _make_updater(
        inlet,
        marker_inlet,
        s_rate,
        channels,
        window_samples,
        state,
        ax,
        lines,
        ern_label,
        sos,
        zi,
        detector,
    )

    _ = animation.FuncAnimation(fig, update_fn, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    print("Filter: bandpass 1-45 Hz, notch 50+60 Hz")
    print(f"ERN threshold: {args.threshold} uV")
    print("Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
