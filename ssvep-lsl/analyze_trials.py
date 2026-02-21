"""Analyze EEG recordings: PSD plots for eyes open vs closed.

Applies the standard signal processing chain (bandpass 1-45 Hz,
notch 50+60 Hz) then plots per-channel PSDs and band power
comparisons. Saves figures as PNGs.

Usage:
    uv run python analyze_trials.py
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

DATA_DIR = "."

CONDITIONS = {
    "nogel": {
        "open": "lsl_dump_1771709185_booth_nogel_eyesopen.csv",
        "closed": (
            "lsl_dump_1771709339_booth_nogel_eyesclosed.csv"
        ),
        "label": "No Gel",
    },
    "gel": {
        "open": "lsl_dump_1771709678_booth_gel_eyesopen.csv",
        "closed": (
            "lsl_dump_1771709769_booth_gel_eyesclosed.csv"
        ),
        "label": "With Gel",
    },
}

ALPHA_BAND = (8, 13)
DISPLAY_CHANNELS = range(5)  # ch0-ch4 only

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def load_dump(path: str) -> tuple[np.ndarray, float, int]:
    """Load CSV. Returns (eeg, sample_rate, n_channels)."""
    df = pd.read_csv(f"{DATA_DIR}/{path}")
    timestamps = df["timestamp"].values
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    eeg = df[ch_cols].values
    s_rate = 1.0 / np.median(np.diff(timestamps))
    return eeg, s_rate, len(ch_cols)


def standard_filter(
    eeg: np.ndarray, s_rate: float,
) -> np.ndarray:
    """Apply standard EEG filter: BP 1-45 Hz + notch 50/60 Hz."""
    bp = butter(
        2, [1.0, 45.0], btype="band", fs=s_rate, output="sos",
    )
    n50 = butter(
        2, [49.0, 51.0], btype="bandstop", fs=s_rate, output="sos",
    )
    n60 = butter(
        2, [59.0, 61.0], btype="bandstop", fs=s_rate, output="sos",
    )
    sos = np.vstack([bp, n50, n60])
    return sosfiltfilt(sos, eeg, axis=0)


def welch_psd(
    eeg_1d: np.ndarray, s_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for a single channel."""
    return welch(
        eeg_1d, fs=s_rate, nperseg=min(1024, len(eeg_1d)),
    )


def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: tuple[float, float],
) -> float:
    """Mean power in a frequency band."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(psd[mask].mean())


def plot_condition(cond_key: str, cond: dict) -> None:
    """Plot PSDs for one gel/no-gel condition."""
    eeg_open, sr_o, n_ch = load_dump(cond["open"])
    eeg_closed, sr_c, _ = load_dump(cond["closed"])

    filt_open = standard_filter(eeg_open, sr_o)
    filt_closed = standard_filter(eeg_closed, sr_c)

    label = cond["label"]
    chs = list(DISPLAY_CHANNELS)
    n_disp = len(chs)

    # --- Figure 1: Filtered PSD per channel ---
    fig, axes = plt.subplots(
        1, n_disp, figsize=(4 * n_disp, 4),
        sharex=True, sharey=True,
    )

    fig.suptitle(
        f"{label} — Filtered PSD (BP 1-45 Hz, Notch 50/60 Hz):"
        " Eyes Open vs Closed",
        fontsize=13, fontweight="bold",
    )

    for i, ch in enumerate(chs):
        ax = axes[i]

        f_o, psd_o = welch_psd(filt_open[:, ch], sr_o)
        f_c, psd_c = welch_psd(filt_closed[:, ch], sr_c)

        ax.semilogy(f_o, psd_o, label="Open", alpha=0.8)
        ax.semilogy(f_c, psd_c, label="Closed", alpha=0.8)
        ax.axvspan(*ALPHA_BAND, alpha=0.1, color="orange")
        ax.set_xlim(0, 50)
        ax.set_title(f"ch{ch}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("Power (log)")
            ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"psd_filtered_{cond_key}.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

    # --- Figure 2: Raw vs Filtered average ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True,
    )
    fig.suptitle(
        f"{label} — Average PSD Across All Channels",
        fontsize=14, fontweight="bold",
    )

    # Raw average (display channels only)
    f_o, psd_o = welch(
        eeg_open[:, chs], fs=sr_o,
        nperseg=min(1024, len(eeg_open)), axis=0,
    )
    f_c, psd_c = welch(
        eeg_closed[:, chs], fs=sr_c,
        nperseg=min(1024, len(eeg_closed)), axis=0,
    )
    ax1.semilogy(f_o, psd_o.mean(axis=1), label="Open")
    ax1.semilogy(f_c, psd_c.mean(axis=1), label="Closed")
    ax1.axvspan(*ALPHA_BAND, alpha=0.1, color="orange")
    ax1.set_xlim(0, 50)
    ax1.set_title("Raw")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power (log)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Filtered average (display channels only)
    f_o, psd_o = welch(
        filt_open[:, chs], fs=sr_o,
        nperseg=min(1024, len(filt_open)), axis=0,
    )
    f_c, psd_c = welch(
        filt_closed[:, chs], fs=sr_c,
        nperseg=min(1024, len(filt_closed)), axis=0,
    )
    ax2.semilogy(f_o, psd_o.mean(axis=1), label="Open")
    ax2.semilogy(f_c, psd_c.mean(axis=1), label="Closed")
    ax2.axvspan(*ALPHA_BAND, alpha=0.1, color="orange")
    ax2.set_xlim(0, 50)
    ax2.set_title("After BP 1-45 Hz + Notch 50/60 Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"psd_avg_{cond_key}.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

    # --- Figure 3: Per-channel band power ratios ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"{label} — Band Power Ratio (Closed / Open) "
        "per Channel",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(n_disp)
    width = 0.18
    band_names = list(BANDS.keys())
    for i, bname in enumerate(band_names):
        lo, hi = BANDS[bname]
        ratios = []
        for ch in chs:
            f_o, psd_o = welch_psd(filt_open[:, ch], sr_o)
            f_c, psd_c = welch_psd(filt_closed[:, ch], sr_c)
            pw_o = band_power(f_o, psd_o, (lo, hi))
            pw_c = band_power(f_c, psd_c, (lo, hi))
            ratios.append(pw_c / pw_o if pw_o > 0 else 0)
        offset = (i - len(band_names) / 2 + 0.5) * width
        colour = "orange" if bname == "alpha" else None
        ax.bar(
            x + offset, ratios, width,
            label=bname, color=colour, alpha=0.8,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{ch}" for ch in chs])
    ax.set_ylabel("Power Ratio (Closed / Open)")
    ax.set_xlabel("Channel")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = f"psd_bands_{cond_key}.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

    # --- Print band power summary ---
    print(f"\n  Band powers (filtered, ch0-ch4 avg):")
    print(
        f"  {'Band':<8s} {'Open':>10s} {'Closed':>10s} "
        f"{'Ratio':>8s}"
    )
    print(f"  {'-' * 40}")
    f_o, psd_o = welch(
        filt_open[:, chs], fs=sr_o,
        nperseg=min(1024, len(filt_open)), axis=0,
    )
    f_c, psd_c = welch(
        filt_closed[:, chs], fs=sr_c,
        nperseg=min(1024, len(filt_closed)), axis=0,
    )
    for name, (lo, hi) in BANDS.items():
        pw_o = band_power(f_o, psd_o.mean(axis=1), (lo, hi))
        pw_c = band_power(f_c, psd_c.mean(axis=1), (lo, hi))
        ratio = pw_c / pw_o if pw_o > 0 else 0
        print(
            f"  {name:<8s} {pw_o:10.4f} {pw_c:10.4f} "
            f"{ratio:7.2f}x"
        )


def main() -> None:
    print("EEG PSD Analysis")
    print("Filter: bandpass 1-45 Hz + notch 50/60 Hz\n")

    for key, cond in CONDITIONS.items():
        print(f"{'=' * 50}")
        print(f"  {cond['label'].upper()}")
        print(f"{'=' * 50}")
        plot_condition(key, cond)
        print()

    print("Done. Open the PNG files to inspect PSDs.")


if __name__ == "__main__":
    main()
