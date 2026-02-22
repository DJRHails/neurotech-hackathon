"""Replay ERP CORE Flanker EEG data as an LSL stream.

Downloads the MNE ERP CORE dataset (single subject, Flanker task),
extracts frontal channels (FCz, Fz, Cz), downsamples to 250 Hz,
and streams the data at real-time pace. Event annotations
(correct/incorrect responses) are printed to the console.
Loops continuously until interrupted.

Usage:
    uv run python test_source_ern.py
    uv run python test_source_ern.py --speed 2.0
"""

import argparse
import time

import mne
import numpy as np
from loguru import logger
from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = "ern_eeg"
TARGET_SRATE = 250
CHUNK_SIZE = 8
CHANNELS = ["FCz", "Fz", "Cz"]


def _load_flanker_data() -> tuple[np.ndarray, list[dict], float]:
    """Load ERP CORE Flanker data and extract frontal channels.

    Returns:
        (data, events, original_srate) where data is (n_samples, n_ch),
        events is a list of {sample, description} dicts, and
        original_srate is the original sampling rate before resampling.
    """
    logger.info("Loading ERP CORE Flanker dataset (may download ~90 MB)...")
    data_path = mne.datasets.erp_core.data_path()
    raw_path = data_path / "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    original_srate = raw.info["sfreq"]
    logger.info(
        "Loaded: {} channels @ {} Hz, {:.1f}s duration",
        len(raw.ch_names),
        original_srate,
        raw.times[-1],
    )

    available = [ch for ch in CHANNELS if ch in raw.ch_names]
    if not available:
        msg = f"None of {CHANNELS} found in dataset. Available: {raw.ch_names[:10]}..."
        raise RuntimeError(msg)
    raw.pick(available)
    logger.info("Using channels: {}", available)

    # Extract and classify response events.
    # Stimulus annotations contain target direction; response annotations
    # contain the actual direction. A mismatch = error (ERN expected).
    events_list = []
    last_target_dir = None
    for ann in raw.annotations:
        desc = ann["description"]
        if "stimulus" in desc and "target_" in desc:
            if "target_left" in desc:
                last_target_dir = "left"
            elif "target_right" in desc:
                last_target_dir = "right"
        elif desc.startswith("response/"):
            resp_dir = desc.split("/")[-1]
            correct = resp_dir == last_target_dir
            label = "correct" if correct else "ERROR"
            sample_idx = int(ann["onset"] * original_srate)
            events_list.append(
                {
                    "sample": sample_idx,
                    "description": f"response/{label} ({resp_dir})",
                    "onset_sec": ann["onset"],
                }
            )
    n_errors = sum(1 for e in events_list if "ERROR" in e["description"])
    logger.info(
        "Found {} response events ({} errors)",
        len(events_list),
        n_errors,
    )

    if original_srate != TARGET_SRATE:
        logger.info(
            "Resampling {} Hz -> {} Hz",
            original_srate,
            TARGET_SRATE,
        )
        raw.resample(TARGET_SRATE, verbose=False)
        # Rescale event sample indices
        ratio = TARGET_SRATE / original_srate
        for ev in events_list:
            ev["sample"] = int(ev["sample"] * ratio)

    data = raw.get_data().T  # (n_samples, n_channels)
    data *= 1e6  # V -> uV
    logger.info(
        "Ready: {} samples, {} channels @ {} Hz",
        data.shape[0],
        data.shape[1],
        TARGET_SRATE,
    )
    return data, events_list, original_srate


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ERP CORE Flanker data as LSL stream")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    args = parser.parse_args()

    data, events, _orig_sr = _load_flanker_data()
    n_samples, n_ch = data.shape

    # Build event lookup: sample_idx -> description
    event_lookup: dict[int, str] = {}
    for ev in events:
        event_lookup[ev["sample"]] = ev["description"]

    info = StreamInfo(
        name=STREAM_NAME,
        type="EEG",
        channel_count=n_ch,
        nominal_srate=TARGET_SRATE,
        source_id="erp_core_flanker",
    )
    ch_xml = info.desc().append_child("channels")
    for ch_name in CHANNELS[:n_ch]:
        ch_xml.append_child("channel").append_child_value("label", ch_name)

    outlet = StreamOutlet(info, chunk_size=CHUNK_SIZE)
    logger.info(
        "Streaming '{}': {} ch ({}) @ {} Hz, speed={:.1f}x",
        STREAM_NAME,
        n_ch,
        CHANNELS[:n_ch],
        TARGET_SRATE,
        args.speed,
    )
    logger.info("Ctrl+C to stop")

    dt = CHUNK_SIZE / TARGET_SRATE / args.speed
    loop_count = 0

    try:
        while True:
            loop_count += 1
            logger.info(
                "--- Loop {} ({} samples, {:.1f}s) ---",
                loop_count,
                n_samples,
                n_samples / TARGET_SRATE,
            )
            idx = 0
            while idx + CHUNK_SIZE <= n_samples:
                chunk = data[idx : idx + CHUNK_SIZE]
                outlet.push_chunk(chunk.tolist())

                # Check for events in this chunk
                for s in range(idx, idx + CHUNK_SIZE):
                    if s in event_lookup:
                        logger.info(
                            "[EVENT] sample={} | {}",
                            s,
                            event_lookup[s],
                        )

                idx += CHUNK_SIZE
                time.sleep(dt)

    except KeyboardInterrupt:
        logger.info("Stopped after {} loops.", loop_count)


if __name__ == "__main__":
    main()
