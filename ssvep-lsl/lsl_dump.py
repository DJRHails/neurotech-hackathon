"""Dump raw LSL stream data to CSV with live visualisation.

Shows per-channel filtered waveforms and a shared PSD panel.
Applies bandpass 1-45 Hz + notch 50/60 Hz by default.

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

from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfilt, sosfilt_zi, welch

PLOT_SECONDS = 5
PSD_UPDATE_INTERVAL = 0.5


def build_filter(
    fs: float,
    n_ch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the standard EEG filter chain.

    Bandpass 1-45 Hz, notch 50 Hz, notch 60 Hz.
    Returns (sos, zi) with zi shaped for n_ch channels.
    """
    bp = butter(
        2,
        [1.0, 45.0],
        btype="band",
        fs=fs,
        output="sos",
    )
    n50 = butter(
        2,
        [49.0, 51.0],
        btype="bandstop",
        fs=fs,
        output="sos",
    )
    n60 = butter(
        2,
        [59.0, 61.0],
        btype="bandstop",
        fs=fs,
        output="sos",
    )
    sos = np.vstack([bp, n50, n60])
    zi_single = sosfilt_zi(sos)  # (n_sections, 2)
    zi = np.repeat(zi_single[:, :, np.newaxis], n_ch, axis=2)
    return sos, zi


def apply_filter(
    data: np.ndarray,
    sos: np.ndarray,
    zi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply IIR filter chain. data: (n_samples, n_ch)."""
    filtered, zi = sosfilt(sos, data, axis=0, zi=zi)
    return filtered.astype(np.float32), zi


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
    n_ch = info.channel_count()
    sr = int(info.nominal_srate())
    print(f"Connected: {info.name()}, {n_ch} channels @ {sr} Hz")

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
        plot = _init_plot(n_ch, sr, buf_len)

    last_psd_update = 0.0

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + [f"ch{i}" for i in range(n_ch)])
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
                        "BP 1-45Hz + Notch 50/60Hz" if use_filter else "Raw (no filter)"
                    )
                    plot["fig"].suptitle(
                        f"LSL: {args.stream} | "
                        f"{n_ch}ch @ {sr}Hz | "
                        f"{count:,} samples | "
                        f"{filt_str}",
                        fontsize=10,
                    )

                plt.pause(0.001)

        except KeyboardInterrupt:
            print(f"\nDone. {count} samples → {output}")
        finally:
            if not args.no_plot:
                plt.close("all")


def _init_plot(
    n_ch: int,
    sr: int,
    buf_len: int,
) -> dict:
    """Set up per-channel waveforms + shared PSD panel."""
    plt.ion()

    # n_ch rows for channels + 1 row for PSD
    n_rows = n_ch + 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 2 * n_rows),
        gridspec_kw={
            "height_ratios": [1] * n_ch + [2],
        },
        sharex=False,
    )

    colours = plt.cm.tab10(np.linspace(0, 1, n_ch))

    ch_axes = []
    ch_lines = []
    for ch in range(n_ch):
        ax = axes[ch]
        (line,) = ax.plot(
            [],
            [],
            linewidth=0.6,
            color=colours[ch],
        )
        ax.set_ylabel(
            f"ch{ch}",
            fontsize=8,
            rotation=0,
            labelpad=25,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(
            axis="both",
            labelsize=7,
        )
        if ch < n_ch - 1:
            ax.tick_params(
                axis="x",
                labelbottom=False,
            )
        else:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_xlim(0, buf_len / sr)
        ch_axes.append(ax)
        ch_lines.append(line)

    # PSD panel at bottom
    ax_psd = axes[n_ch]
    psd_lines = []
    for ch in range(n_ch):
        (line,) = ax_psd.semilogy(
            [],
            [],
            linewidth=1.0,
            color=colours[ch],
            label=f"ch{ch}",
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

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.show()

    return {
        "fig": fig,
        "ch_axes": ch_axes,
        "ch_lines": ch_lines,
        "ax_psd": ax_psd,
        "psd_lines": psd_lines,
    }


if __name__ == "__main__":
    main()
