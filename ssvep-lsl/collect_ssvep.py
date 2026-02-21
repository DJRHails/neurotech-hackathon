"""Collect labeled SSVEP training data.

Shows flickering checkerboard stimuli while recording EEG via LSL.
Press 1-4 to start a labeled trial for the corresponding target.
Each trial records `signal_len` seconds of EEG and appends to a
per-class CSV file.

Usage:
    uv run python collect_ssvep.py -s "lock_in_eeg_processed"
    uv run python collect_ssvep.py --suffix session1
"""

import argparse
import csv
import threading
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from psychopy import event, visual
from pylsl import StreamInlet, resolve_byprop
from ssvep import CheckerBoard

TARGET_NAMES = ["bottom_left", "top_left", "top_right", "bottom_right"]
TARGET_KEYS = ["1", "2", "3", "4"]
TARGET_POS = [(-0.6, -0.6), (-0.6, 0.6), (0.6, 0.6), (0.6, -0.6)]
TARGET_LABELS = ["\u2199", "\u2196", "\u2197", "\u2198"]
FRAME_RATES = [5, 6, 7, 8]


def _lsl_reader(
    inlet: StreamInlet,
    buf: list[list[float]],
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    """Background thread: pull LSL samples into shared buffer."""
    while not stop.is_set():
        chunk, timestamps = inlet.pull_chunk(
            timeout=0.1, max_samples=32
        )
        if chunk:
            with lock:
                for ts, sample in zip(timestamps, chunk):
                    buf.append([ts, *sample])


def _save_trial(
    rows: list[list[float]],
    n_channels: int,
    path: str,
    trial_num: int,
) -> None:
    """Append one trial's data to a CSV file."""
    write_header = True
    try:
        with open(path) as f:
            write_header = f.readline() == ""
    except FileNotFoundError:
        pass

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["trial", "timestamp"]
                + [f"ch{i}" for i in range(n_channels)]
            )
        for row in rows:
            writer.writerow([trial_num, *row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect labeled SSVEP training data"
    )
    parser.add_argument(
        "-s",
        "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name (default: lock_in_eeg_processed)",
    )
    parser.add_argument(
        "--signal-len",
        type=float,
        default=3.0,
        help="Trial length in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filenames",
    )
    args = parser.parse_args()

    print(f"Resolving LSL stream '{args.stream}'...")
    streams = resolve_byprop(
        "name", args.stream, minimum=1, timeout=10
    )
    if not streams:
        raise RuntimeError(
            f"No LSL stream '{args.stream}' found. "
            "Is the EEG source running?"
        )

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    n_ch = info.channel_count()
    s_rate = int(info.nominal_srate())
    print(f"Connected: {info.name()}, {n_ch} ch @ {s_rate} Hz")

    tag = f"_{args.suffix}" if args.suffix else ""
    output_paths = {
        i: f"ssvep_class_{name}{tag}.csv"
        for i, name in enumerate(TARGET_NAMES)
    }
    trial_counts = {i: 0 for i in range(len(TARGET_NAMES))}

    # Shared EEG buffer
    buf: list[list[float]] = []
    lock = threading.Lock()
    stop = threading.Event()
    reader = threading.Thread(
        target=_lsl_reader,
        args=(inlet, buf, lock, stop),
        daemon=True,
    )
    reader.start()

    # Display
    win = visual.Window(
        [800, 600],
        monitor="testMonitor",
        fullscr=True,
        screen=1,
        units="norm",
        color=[0.1, 0.1, 0.1],
    )
    aspect = win.size[1] / win.size[0]
    stim_size = (0.6 * aspect, 0.6)

    targets: list[CheckerBoard] = []
    for fr, pos in zip(FRAME_RATES, TARGET_POS):
        targets.append(
            CheckerBoard(
                window=win,
                size=stim_size,
                n_frame=fr,
                position=pos,
                log_time=False,
            )
        )

    key_labels = [
        visual.TextStim(
            win=win, pos=pos, text=f"[{key}]",
            color=(0.6, 0.6, 0.6), height=0.08,
        )
        for key, pos in zip(TARGET_KEYS, TARGET_POS)
    ]

    status_text = visual.TextStim(
        win=win, pos=(0, 0),
        text="Press 1-4 to record a trial\nESC to quit",
        color=(1, 1, 1), height=0.08,
    )

    recording_target: int | None = None
    recording_start: float = 0.0

    print("Press 1-4 to record trials. ESC to quit.")
    print(f"Trial length: {args.signal_len}s")
    for i, path in output_paths.items():
        print(f"  [{i + 1}] {TARGET_NAMES[i]} → {path}")

    try:
        while True:
            win.flip()
            for stim in targets:
                stim.draw()
            for label in key_labels:
                label.draw()

            if recording_target is not None:
                elapsed = time.time() - recording_start
                remaining = args.signal_len - elapsed
                idx = recording_target
                status_text.text = (
                    f"Recording {TARGET_NAMES[idx]}... "
                    f"{remaining:.1f}s"
                )
                status_text.draw()

                # Highlight active target
                highlight = visual.Circle(
                    win=win,
                    radius=0.35,
                    pos=TARGET_POS[idx],
                    lineColor=(0, 1, 0),
                    lineWidth=4,
                    fillColor=None,
                )
                highlight.draw()

                if elapsed >= args.signal_len:
                    with lock:
                        trial_data = list(buf)
                        buf.clear()
                    trial_counts[idx] += 1
                    _save_trial(
                        trial_data,
                        n_ch,
                        output_paths[idx],
                        trial_counts[idx],
                    )
                    n_saved = len(trial_data)
                    print(
                        f"  Saved trial {trial_counts[idx]} "
                        f"for {TARGET_NAMES[idx]} "
                        f"({n_saved} samples)"
                    )
                    recording_target = None
            else:
                status_text.text = (
                    "Press 1-4 to record a trial\nESC to quit"
                )
                status_text.draw()

            keys = event.getKeys()
            if "escape" in keys:
                break

            if recording_target is None:
                for i, key in enumerate(TARGET_KEYS):
                    if key in keys:
                        recording_target = i
                        recording_start = time.time()
                        with lock:
                            buf.clear()
                        print(
                            f"Recording target {i + 1} "
                            f"({TARGET_NAMES[i]})..."
                        )
                        break
    finally:
        win.close()
        stop.set()
        reader.join(timeout=2)
        inlet.close_stream()

    print("\nCollection summary:")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name}: {trial_counts[i]} trials → {output_paths[i]}")


if __name__ == "__main__":
    main()
