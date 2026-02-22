"""Dump raw LSL stream data to CSV with live visualisation.

Shows two windows: per-channel filtered waveforms, and a PSD plot
with one line per channel. Applies bandpass 1-45 Hz + notch 50/60 Hz
by default.

Usage:
    uv run python lsl_dump.py
    uv run python lsl_dump.py -s unicorn_eeg --suffix baseline
    uv run python lsl_dump.py --no-plot --no-filter
"""

import argparse
import csv
import time
from collections import deque

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from filters import apply_filter, build_filter
from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch

PLOT_SECONDS = 5
PSD_UPDATE_INTERVAL = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump LSL stream to CSV with live display"
    )
    parser.add_argument(
        "-s",
        "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name (default: lock_in_eeg_processed)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filename",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable live plotting",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip bandpass/notch filtering",
    )
    parser.add_argument(
        "-c",
        "--channels",
        nargs="*",
        type=int,
        default=list(range(8)),
        help="Channel indices to record (default: 0-7)",
    )
    args = parser.parse_args()

    tag = f"_{args.suffix}" if args.suffix else ""
    output = f"lsl_dump_{int(time.time())}{tag}.csv"

    print(f"Resolving LSL stream '{args.stream}'...")
    streams = resolve_byprop("name", args.stream, minimum=1, timeout=10)
    if not streams:
        print(f"No stream '{args.stream}' found.")
        raise SystemExit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    total_ch = info.channel_count()
    sr = int(info.nominal_srate())
    channels = args.channels
    bad = [c for c in channels if c >= total_ch]
    if bad:
        print(
            f"Channels {bad} out of range "
            f"(stream has {total_ch})"
        )
        raise SystemExit(1)
    n_ch = len(channels)
    print(
        f"Connected: {info.name()}, "
        f"{total_ch} ch @ {sr} Hz "
        f"(using {n_ch}: {channels})"
    )

    use_filter = not args.no_filter
    if use_filter:
        sos, zi = build_filter(float(sr), n_ch)
        print("Filter: bandpass 1-45 Hz, notch 50+60 Hz")
    else:
        print("Filter: disabled")

    print(f"Writing to {output}  (Ctrl+C to stop)")

    buf_len = PLOT_SECONDS * sr
    ring: deque[list[float]] = deque(maxlen=buf_len)

    if not args.no_plot:
        plot = _init_plot(channels, sr, buf_len)

    last_psd_update = 0.0

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp"] + [f"ch{i}" for i in channels]
        )
        count = 0
        try:
            while True:
                chunk, timestamps = inlet.pull_chunk(
                    timeout=0.05,
                    max_samples=64,
                )
                if not chunk:
                    if not args.no_plot:
                        plt.pause(0.01)
                    continue

                samples = np.array(chunk, dtype=np.float32)
                samples = samples[:, channels]

                if use_filter:
                    samples, zi = apply_filter(
                        samples,
                        sos,
                        zi,
                    )

                for i, ts_val in enumerate(timestamps):
                    row = samples[i].tolist()
                    writer.writerow([ts_val] + row)
                    ring.append(row)
                    count += 1

                if count % 250 == 0:
                    f.flush()

                if args.no_plot:
                    if count % (sr * 5) == 0:
                        print(f"  {count} samples...")
                    continue

                # Update per-channel traces every chunk
                if len(ring) > 10:
                    data = np.array(ring)
                    t = np.arange(len(data)) / sr
                    for ch in range(n_ch):
                        plot["ch_lines"][ch].set_data(
                            t,
                            data[:, ch],
                        )
                        plot["ch_axes"][ch].set_xlim(
                            0,
                            t[-1],
                        )
                        plot["ch_axes"][ch].relim()
                        plot["ch_axes"][ch].autoscale_view(
                            scalex=False,
                            scaley=True,
                        )

                # Update PSD periodically
                now = time.monotonic()
                if now - last_psd_update > PSD_UPDATE_INTERVAL and len(ring) >= sr:
                    last_psd_update = now
                    data = np.array(ring)
                    nperseg = min(512, len(data))
                    for ch in range(n_ch):
                        freqs, psd_vals = welch(
                            data[:, ch],
                            fs=sr,
                            nperseg=nperseg,
                        )
                        plot["psd_lines"][ch].set_data(
                            freqs,
                            psd_vals,
                        )
                    plot["ax_psd"].relim()
                    plot["ax_psd"].autoscale_view()
                    plot["ax_psd"].set_xlim(0, 50)

                    filt_str = (
                        "BP 1-45Hz + Notch 50/60Hz"
                        if use_filter
                        else "Raw (no filter)"
                    )
                    title = (
                        f"LSL: {args.stream} | "
                        f"{n_ch}ch @ {sr}Hz | "
                        f"{count:,} samples | "
                        f"{filt_str}"
                    )
                    plot["fig_raw"].suptitle(
                        title, fontsize=10
                    )
                    plot["fig_psd"].suptitle(
                        title, fontsize=10
                    )

                plt.pause(0.001)

        except KeyboardInterrupt:
            print(f"\nDone. {count} samples → {output}")
        finally:
            if not args.no_plot:
                plt.close("all")


def _init_plot(
    channels: list[int],
    sr: int,
    buf_len: int,
) -> dict:
    """Set up two windows: raw channels and PSD."""
    n_ch = len(channels)
    plt.ion()
    colours = plt.cm.tab10(np.linspace(0, 1, n_ch))

    # Window 1: raw channel waveforms
    fig_raw, axes_raw = plt.subplots(
        n_ch,
        1,
        figsize=(14, 2 * n_ch),
        sharex=True,
        num="Raw Channels",
    )
    if n_ch == 1:
        axes_raw = [axes_raw]

    ch_axes = []
    ch_lines = []
    for idx, ch_id in enumerate(channels):
        ax = axes_raw[idx]
        (line,) = ax.plot(
            [],
            [],
            linewidth=0.6,
            color=colours[idx],
        )
        ax.set_ylabel(
            f"ch{ch_id}",
            fontsize=8,
            rotation=0,
            labelpad=25,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(0, buf_len / sr)
        ch_axes.append(ax)
        ch_lines.append(line)

    axes_raw[-1].set_xlabel("Time (s)", fontsize=8)
    fig_raw.tight_layout(rect=[0, 0, 1, 0.96])
    fig_raw.canvas.draw()
    fig_raw.show()

    # Window 2: PSD (one line per channel)
    fig_psd, ax_psd = plt.subplots(
        1,
        1,
        figsize=(10, 5),
        num="PSD",
    )
    psd_lines = []
    for idx, ch_id in enumerate(channels):
        (line,) = ax_psd.semilogy(
            [],
            [],
            linewidth=1.0,
            color=colours[idx],
            label=f"ch{ch_id}",
            alpha=0.8,
        )
        psd_lines.append(line)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_psd.set_ylabel("Power", fontsize=9)
    ax_psd.set_title("PSD (Welch)", fontsize=9)
    ax_psd.legend(
        loc="upper right",
        fontsize=7,
        ncol=n_ch,
    )
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_xlim(0, 50)
    fig_psd.tight_layout()
    fig_psd.canvas.draw()
    fig_psd.show()

    return {
        "fig_raw": fig_raw,
        "fig_psd": fig_psd,
        "ch_axes": ch_axes,
        "ch_lines": ch_lines,
        "ax_psd": ax_psd,
        "psd_lines": psd_lines,
    }


if __name__ == "__main__":
    main()
