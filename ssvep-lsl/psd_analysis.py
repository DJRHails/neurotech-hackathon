"""PSD analysis for EEG recordings.

Loads LSL dump CSVs and Unicorn recorder CSVs, computes
Welch PSD per channel, and saves per-file summary plots.

Usage:
    uv run python psd_analysis.py
    uv run python psd_analysis.py --files lsl_dump_*.csv
    uv run python psd_analysis.py --max-freq 60
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.signal import welch

UNICORN_SRATE = 250
LSL_SRATE = 250

CHANNEL_LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

BAND_COLORS = {
    "delta": "#4C72B0",
    "theta": "#55A868",
    "alpha": "#C44E52",
    "beta": "#8172B2",
    "gamma": "#CCB974",
}


def _ch_columns(header: str) -> list[int]:
    """Return column indices of ch* columns in a CSV header."""
    cols = [c.strip() for c in header.split(",")]
    return [
        i for i, c in enumerate(cols)
        if c.startswith("ch")
    ]


def load_lsl_dump(path: Path) -> tuple[np.ndarray, int]:
    """Load LSL dump CSV. Returns (data, sample_rate)."""
    header = path.read_text().split("\n")[0]
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    ch_cols = _ch_columns(header)
    if ch_cols:
        eeg = raw[:, ch_cols]
    else:
        eeg = raw[:, 1:]
    return eeg, LSL_SRATE


def load_unicorn(path: Path) -> tuple[np.ndarray, int]:
    """Load Unicorn Recorder CSV. Returns (data, sample_rate)."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    eeg = raw[:, :8]
    return eeg, UNICORN_SRATE


def load_ssvep_class(path: Path) -> tuple[np.ndarray, int]:
    """Load SSVEP classification CSV. Returns (data, srate)."""
    header = path.read_text().split("\n")[0]
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    ch_cols = _ch_columns(header)
    if ch_cols:
        eeg = raw[:, ch_cols]
    else:
        eeg = raw[:, 2:]
    return eeg, LSL_SRATE


def load_csv(path: Path) -> tuple[np.ndarray, int]:
    """Auto-detect format and load."""
    header = path.read_text().split("\n")[0]
    if header.startswith("trial,"):
        return load_ssvep_class(path)
    if header.startswith("timestamp,"):
        return load_lsl_dump(path)
    if "EEG 1" in header:
        return load_unicorn(path)
    msg = f"Unknown CSV format in {path}: {header[:80]}"
    raise ValueError(msg)


def compute_psd(
    eeg: np.ndarray, s_rate: int, max_freq: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for each channel.

    Returns:
        freqs: frequency axis (n_freqs,)
        psd: power array (n_channels, n_freqs)
    """
    n_channels = eeg.shape[1]
    psds = []
    for ch in range(n_channels):
        freqs, pxx = welch(
            eeg[:, ch], fs=s_rate, nperseg=min(s_rate * 4, len(eeg[:, ch])), noverlap=None
        )
        mask = freqs <= max_freq
        psds.append(pxx[mask])

    return freqs[mask], np.array(psds)


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    title: str,
    out_path: Path,
    max_freq: float = 50.0,
) -> None:
    """Plot PSD per channel with band shading."""
    n_channels = psd.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        label = CHANNEL_LABELS[ch_idx] if ch_idx < len(CHANNEL_LABELS) else f"Ch{ch_idx}"
        psd_db = 10 * np.log10(psd[ch_idx] + 1e-20)
        ax.plot(freqs, psd_db, color="k", linewidth=0.8)

        for band_name, (f_lo, f_hi) in BANDS.items():
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            if band_mask.any():
                ax.fill_between(
                    freqs[band_mask],
                    psd_db[band_mask],
                    alpha=0.3,
                    color=BAND_COLORS[band_name],
                    label=band_name,
                )

        ax.set_ylabel(f"{label}\n(dB/Hz)")
        ax.set_xlim(0, max_freq)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncol=len(BANDS), fontsize=8)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: {}", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PSD analysis for EEG recordings")
    parser.add_argument(
        "--files", nargs="*", help="Specific CSV files (default: all in . and data/)"
    )
    parser.add_argument(
        "--max-freq", type=float, default=50.0, help="Max frequency for PSD plot (default: 50 Hz)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="plots", help="Output directory for plots (default: plots/)"
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    out_dir = base / args.out_dir
    out_dir.mkdir(exist_ok=True)

    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        csv_files = sorted(
            list(base.glob("lsl_dump_*.csv"))
            + list(base.glob("ssvep_class_*.csv"))
            + list((base / "data").glob("*.csv"))
        )

    if not csv_files:
        logger.warning("No CSV files found.")
        return

    logger.info("Processing {} files...", len(csv_files))

    for path in csv_files:
        logger.info("[{}]", path.name)
        try:
            eeg, s_rate = load_csv(path)
        except ValueError as exc:
            logger.warning("  SKIP: {}", exc)
            continue

        n_samples, n_ch = eeg.shape
        duration = n_samples / s_rate
        logger.info("  {} channels, {} samples, {:.1f}s @ {} Hz", n_ch, n_samples, duration, s_rate)

        freqs, psd = compute_psd(eeg, s_rate, max_freq=args.max_freq)
        plot_psd(
            freqs,
            psd,
            title=path.stem,
            out_path=out_dir / f"{path.stem}_psd.png",
            max_freq=args.max_freq,
        )


if __name__ == "__main__":
    main()
