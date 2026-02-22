"""Eriksen Flanker Task with ERN recording via LSL.

Presents congruent/incongruent arrow stimuli and records
response-locked EEG epochs for ERN analysis. Subjects respond
to the central arrow direction while ignoring flanking arrows.

Usage:
    uv run python flanker_task.py -s "lock_in_eeg_processed"
    uv run python flanker_task.py -s None -n 10   # stimulus-only
"""

import argparse
import csv
import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from psychopy import core, event, visual  # noqa: E402
from pylsl import StreamInlet, resolve_byprop  # noqa: E402

from ern_detector import ERNDetector  # noqa: E402
from filters import apply_filter, build_filter  # noqa: E402

FIXATION_MS = 500
STIMULUS_TIMEOUT_MS = 1500
FEEDBACK_MS = 500
ITI_MIN_MS = 800
ITI_MAX_MS = 1200

# EEG epoch around response: 200ms before to 500ms after
EPOCH_PRE_MS = 200
EPOCH_POST_MS = 500
MIN_EPOCH_SAMPLES = 10

ARROW_LEFT = "left"
ARROW_RIGHT = "right"
RESPONSE_KEYS = [ARROW_LEFT, ARROW_RIGHT]

CONGRUENT_STIMULI = {
    ARROW_LEFT: "< < < < <",
    ARROW_RIGHT: "> > > > >",
}
INCONGRUENT_STIMULI = {
    ARROW_LEFT: "> > < > >",
    ARROW_RIGHT: "< < > < <",
}


def _lsl_reader(
    inlet: StreamInlet,
    push_fn: callable,
    stop: threading.Event,
    channels: list[int] | None = None,
) -> None:
    """Background thread: pull LSL chunks into the task."""
    while not stop.is_set():
        chunk, _ = inlet.pull_chunk(
            timeout=0.1,
            max_samples=32,
        )
        if chunk:
            samples = np.array(chunk, dtype=np.float32)
            if channels is not None:
                samples = samples[:, channels]
            push_fn(samples)


def _build_trial_sequence(n_trials: int) -> list[dict]:
    """Build a balanced, randomised trial sequence.

    Returns list of dicts with keys: condition, direction.
    Half congruent, half incongruent, fully shuffled.
    """
    trials = []
    for i in range(n_trials):
        condition = "congruent" if i % 2 == 0 else "incongruent"
        direction = ARROW_LEFT if i % 4 < 2 else ARROW_RIGHT  # noqa: PLR2004
        trials.append(
            {
                "condition": condition,
                "direction": direction,
            }
        )
    random.shuffle(trials)
    return trials


class FlankerTask:
    """Eriksen flanker task with response-locked EEG epochs.

    Args:
        s_rate: EEG sampling rate in Hz.
        n_channels: Number of EEG channels.
        n_trials: Total number of trials.
        threshold_uv: ERN detection threshold in uV.
        use_lsl: Whether EEG recording is active.
    """

    def __init__(
        self,
        s_rate: int,
        n_channels: int,
        n_trials: int,
        threshold_uv: float = -5.0,
        use_lsl: bool = True,
    ) -> None:
        self._s_rate = s_rate
        self._n_channels = n_channels
        self._n_trials = n_trials
        self._use_lsl = use_lsl
        self._lock = threading.Lock()
        self._eeg_buf: np.ndarray = np.empty(
            (0, n_channels),
            dtype=np.float32,
        )
        self._eeg_timestamps: list[float] = []

        if use_lsl:
            self._sos, self._zi = build_filter(
                float(s_rate),
                n_channels,
            )
            self._detector = ERNDetector(
                s_rate=s_rate,
                threshold_uv=threshold_uv,
            )

        self._results: list[dict] = []
        self._epoch_pre = int(EPOCH_PRE_MS * s_rate / 1000)
        self._epoch_post = int(EPOCH_POST_MS * s_rate / 1000)

    def push_eeg(self, samples: np.ndarray) -> None:
        """Thread-safe: accept new EEG samples from LSL reader."""
        with self._lock:
            now = time.time()
            if self._eeg_buf.shape[0] == 0:
                self._eeg_buf = samples
            else:
                self._eeg_buf = np.concatenate(
                    (self._eeg_buf, samples),
                    axis=0,
                )
            dt = 1.0 / self._s_rate
            for i in range(samples.shape[0]):
                self._eeg_timestamps.append(now - (samples.shape[0] - 1 - i) * dt)

    def run(self, duration: float) -> list[dict]:
        """Run the flanker task. Returns list of trial results."""
        win = visual.Window(
            [800, 600],
            monitor="testMonitor",
            fullscr=True,
            screen=1,
            units="norm",
            color=[0.1, 0.1, 0.1],
        )

        fixation = visual.TextStim(
            win,
            text="+",
            height=0.15,
            color="white",
        )
        stim_text = visual.TextStim(
            win,
            text="",
            height=0.2,
            color="white",
            font="Courier New",
        )
        feedback_text = visual.TextStim(
            win,
            text="",
            height=0.12,
            color="white",
        )
        info_text = visual.TextStim(
            win,
            text="",
            pos=(0, -0.8),
            height=0.05,
            color="gray",
        )

        trials = _build_trial_sequence(self._n_trials)
        clock = core.Clock()
        start_time = time.time()

        try:
            for trial_num, trial in enumerate(trials, 1):
                if time.time() - start_time > duration:
                    break

                result = self._run_single_trial(
                    win,
                    fixation,
                    stim_text,
                    feedback_text,
                    info_text,
                    clock,
                    trial_num,
                    trial,
                )
                self._results.append(result)

                self._print_trial_result(trial_num, result)

                if "escape" in event.getKeys():
                    break
        finally:
            win.close()

        return self._results

    def _run_single_trial(
        self,
        win: visual.Window,
        fixation: visual.TextStim,
        stim_text: visual.TextStim,
        feedback_text: visual.TextStim,
        info_text: visual.TextStim,
        clock: core.Clock,
        trial_num: int,
        trial: dict,
    ) -> dict:
        """Execute one trial: fixation -> stimulus -> feedback -> ITI."""
        condition = trial["condition"]
        direction = trial["direction"]
        stimuli = CONGRUENT_STIMULI if condition == "congruent" else INCONGRUENT_STIMULI
        arrow_str = stimuli[direction]

        info_text.text = f"Trial {trial_num}/{self._n_trials}"

        # Fixation
        self._show_for_duration(
            win,
            [fixation, info_text],
            FIXATION_MS / 1000,
        )

        # Stimulus presentation + response collection
        event.clearEvents()
        stim_text.text = arrow_str
        clock.reset()
        response = None
        rt_ms = None

        deadline = time.time() + STIMULUS_TIMEOUT_MS / 1000
        while time.time() < deadline:
            stim_text.draw()
            info_text.draw()
            win.flip()
            keys = event.getKeys(
                keyList=[*RESPONSE_KEYS, "escape"],
                timeStamped=clock,
            )
            if keys:
                key, rt = keys[0]
                if key == "escape":
                    event.clearEvents()
                    return self._make_result(
                        trial_num,
                        condition,
                        direction,
                        None,
                        None,
                        None,
                    )
                response = key
                rt_ms = rt * 1000
                break

        response_time = time.time()
        correct = response == direction if response else None

        # Feedback
        if response is None:
            feedback_text.text = "TOO SLOW"
            feedback_text.color = "yellow"
        elif correct:
            feedback_text.text = "CORRECT"
            feedback_text.color = "#81C784"
        else:
            feedback_text.text = "ERROR"
            feedback_text.color = "#FF5252"

        self._show_for_duration(
            win,
            [feedback_text, info_text],
            FEEDBACK_MS / 1000,
        )

        # Extract epoch and detect ERN
        epoch_data = None
        ern_detected = False
        if self._use_lsl and response is not None:
            epoch_data, ern_detected = self._extract_epoch(response_time)

        # Inter-trial interval (jittered)
        iti = random.uniform(  # noqa: S311
            ITI_MIN_MS / 1000,
            ITI_MAX_MS / 1000,
        )
        self._show_for_duration(win, [info_text], iti)

        return self._make_result(
            trial_num,
            condition,
            direction,
            response,
            correct,
            rt_ms,
            ern_detected,
            epoch_data,
        )

    def _extract_epoch(
        self,
        response_time: float,
    ) -> tuple[np.ndarray | None, bool]:
        """Extract EEG epoch around response and run ERN detection."""
        # Wait for post-response data to arrive
        core.wait(EPOCH_POST_MS / 1000 + 0.05)

        with self._lock:
            if len(self._eeg_timestamps) == 0:
                return None, False

            ts = np.array(self._eeg_timestamps)
            pre_start = response_time - EPOCH_PRE_MS / 1000
            post_end = response_time + EPOCH_POST_MS / 1000

            mask = (ts >= pre_start) & (ts <= post_end)
            if mask.sum() < MIN_EPOCH_SAMPLES:
                return None, False

            epoch = self._eeg_buf[mask].copy()

        # Filter the epoch
        epoch_filtered, _ = apply_filter(
            epoch,
            self._sos,
            self._zi,
        )

        # Detect ERN in the post-response portion
        post_mask_start = int(
            self._epoch_pre * epoch_filtered.shape[0] / (self._epoch_pre + self._epoch_post)
        )
        post_data = epoch_filtered[post_mask_start:]
        hits = self._detector.detect(post_data, response_time)

        return epoch_filtered, len(hits) > 0

    def _show_for_duration(
        self,
        win: visual.Window,
        stimuli: list,
        duration_s: float,
    ) -> None:
        """Draw stimuli for a fixed duration."""
        deadline = time.time() + duration_s
        while time.time() < deadline:
            for stim in stimuli:
                stim.draw()
            win.flip()

    def _make_result(
        self,
        trial_num: int,
        condition: str,
        direction: str,
        response: str | None,
        correct: bool | None,
        rt_ms: float | None,
        ern_detected: bool = False,
        epoch_data: np.ndarray | None = None,
    ) -> dict:
        """Build a trial result dict."""
        return {
            "trial": trial_num,
            "condition": condition,
            "direction": direction,
            "response": response or "none",
            "correct": correct,
            "rt_ms": round(rt_ms, 1) if rt_ms else None,
            "timestamp": time.time(),
            "ern_detected": ern_detected,
            "epoch": epoch_data,
        }

    def _print_trial_result(
        self,
        trial_num: int,
        result: dict,
    ) -> None:
        """Print single trial to console."""
        tag = ""
        if result["ern_detected"]:
            tag = " [ERN]"
        correct_str = (
            "correct" if result["correct"] else "error" if result["correct"] is not None else "miss"
        )
        rt_str = f"{result['rt_ms']:.0f}ms" if result["rt_ms"] else "---"
        print(
            f"  T{trial_num:03d} "
            f"{result['condition']:>11s} "
            f"{result['direction']:>5s} "
            f"{correct_str:>7s} "
            f"{rt_str:>6s}{tag}"
        )


def _save_results(
    results: list[dict],
    channels: list[int],
    path: str,
) -> None:
    """Write trial results and EEG epochs to CSV."""
    n_channels = len(channels)
    header = [
        "trial",
        "condition",
        "direction",
        "response",
        "correct",
        "rt_ms",
        "timestamp",
    ] + [f"ch{i}" for i in channels]

    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            base = [
                result["trial"],
                result["condition"],
                result["direction"],
                result["response"],
                result["correct"],
                result["rt_ms"],
                f"{result['timestamp']:.3f}",
            ]

            epoch = result.get("epoch")
            if epoch is not None and epoch.shape[0] > 0:
                for row in epoch:
                    ch_data = [f"{row[i]:.2f}" if i < len(row) else "" for i in range(n_channels)]
                    writer.writerow(base + ch_data)
            else:
                writer.writerow(base + [""] * n_channels)


def _print_summary(results: list[dict]) -> None:
    """Print session summary: accuracy, RT, ERN stats."""
    if not results:
        print("No trials completed.")
        return

    responded = [r for r in results if r["correct"] is not None]
    if not responded:
        print("No valid responses recorded.")
        return

    congruent = [r for r in responded if r["condition"] == "congruent"]
    incongruent = [r for r in responded if r["condition"] == "incongruent"]

    def _acc(trials: list[dict]) -> float:
        if not trials:
            return 0.0
        return sum(1 for t in trials if t["correct"]) / len(trials) * 100

    def _mean_rt(trials: list[dict]) -> float:
        rts = [t["rt_ms"] for t in trials if t["rt_ms"]]
        return np.mean(rts) if rts else 0.0

    errors = [r for r in responded if not r["correct"]]
    correct = [r for r in responded if r["correct"]]
    ern_on_errors = sum(1 for r in errors if r["ern_detected"])
    ern_on_correct = sum(1 for r in correct if r["ern_detected"])

    print("\n--- Session Summary ---")
    print(f"Trials completed: {len(responded)}")
    print(
        f"Congruent:   "
        f"acc={_acc(congruent):.0f}%  "
        f"RT={_mean_rt(congruent):.0f}ms  "
        f"(n={len(congruent)})"
    )
    print(
        f"Incongruent: "
        f"acc={_acc(incongruent):.0f}%  "
        f"RT={_mean_rt(incongruent):.0f}ms  "
        f"(n={len(incongruent)})"
    )
    print(f"Overall:     acc={_acc(responded):.0f}%  RT={_mean_rt(responded):.0f}ms")

    if errors:
        print(
            f"ERN on errors:  "
            f"{ern_on_errors}/{len(errors)} "
            f"({ern_on_errors / len(errors) * 100:.0f}%)"
        )
    if correct:
        print(
            f"ERN on correct: "
            f"{ern_on_correct}/{len(correct)} "
            f"({ern_on_correct / len(correct) * 100:.0f}%)"
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the flanker task."""
    parser = argparse.ArgumentParser(description="Eriksen Flanker Task with ERN recording")
    parser.add_argument(
        "-s",
        "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name, or 'None' for stimulus-only mode",
    )
    parser.add_argument(
        "-n",
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials (default: 100)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=600,
        help="Max duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filename",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=-5.0,
        help="ERN detection threshold in uV (default: -5.0)",
    )
    parser.add_argument(
        "-c",
        "--channels",
        nargs="*",
        type=int,
        default=list(range(8)),
        help="Channel indices to record (default: 0-7)",
    )
    return parser.parse_args()


def _connect_lsl(
    stream_name: str,
    channels: list[int],
) -> tuple[StreamInlet, int, int]:
    """Resolve LSL stream and validate channels.

    Returns:
        (inlet, s_rate, n_channels) tuple.
    """
    print(f"Resolving LSL stream '{stream_name}'...")
    streams = resolve_byprop(
        "name",
        stream_name,
        minimum=1,
        timeout=10,
    )
    if not streams:
        msg = f"No LSL stream '{stream_name}' found. Is the EEG source running?"
        raise RuntimeError(msg)
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    s_rate = int(info.nominal_srate())
    total_ch = info.channel_count()
    bad = [c for c in channels if c >= total_ch]
    if bad:
        msg = f"Channels {bad} out of range (stream has {total_ch})"
        raise RuntimeError(msg)
    n_ch = len(channels)
    print(f"Connected: {info.name()}, {total_ch} ch @ {s_rate} Hz (using {n_ch}: {channels})")
    return inlet, s_rate, n_ch


def main() -> None:
    args = _parse_args()
    use_lsl = args.stream.lower() != "none"
    channels = args.channels

    if use_lsl:
        inlet, s_rate, n_ch = _connect_lsl(
            args.stream,
            channels,
        )
    else:
        s_rate = 250
        n_ch = len(channels)
        print("Stimulus-only mode (no LSL stream)")

    task = FlankerTask(
        s_rate=s_rate,
        n_channels=n_ch,
        n_trials=args.n_trials,
        threshold_uv=args.threshold,
        use_lsl=use_lsl,
    )

    if use_lsl:
        print("Filter: bandpass 1-45 Hz, notch 50+60 Hz")

    stop = threading.Event()
    if use_lsl:
        reader = threading.Thread(
            target=_lsl_reader,
            args=(inlet, task.push_eeg, stop, channels),
            daemon=True,
        )
        reader.start()

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.suffix}" if args.suffix else ""
    output_path = f"flanker_ern_{ts}{tag}.csv"

    print(f"\nStarting flanker task: {args.n_trials} trials")
    print("Respond with LEFT/RIGHT arrow to the CENTRE arrow")
    print("Press ESC to abort\n")

    try:
        results = task.run(args.duration)
    finally:
        stop.set()
        if use_lsl:
            reader.join(timeout=2)
            inlet.close_stream()

    _save_results(results, channels, output_path)
    _print_summary(results)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
