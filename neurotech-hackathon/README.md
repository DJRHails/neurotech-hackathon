# SSVEP BCI — Neurotech Hackathon

Real-time SSVEP (Steady-State Visual Evoked Potential) brain-computer interface using the Unicorn Hybrid Black EEG headset.

Two images flicker at 10 Hz and 15 Hz. The system detects which one you're looking at via FFT power analysis of occipital EEG channels.

## Quick Start (macOS — simulated)

```bash
cd neurotech-hackathon
uv sync
uv run python main.py
```

This uses a simulated 10 Hz sine generator. You should see the left image highlighted (detected as LEFT).

## Cross-Machine Setup (Windows → macOS via LSL)

The Unicorn Hybrid Black requires Windows drivers. Use Lab Streaming Layer (LSL) to stream EEG from Windows to macOS over the network.

### Windows Setup

1. Install Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone and install:
   ```powershell
   cd neurotech-hackathon
   uv sync
   ```

3. Connect the Unicorn Hybrid Black headset via Bluetooth

4. Run the streamer:
   ```powershell
   uv run python windows_stream.py
   ```
   This opens a window showing the raw EEG and streams it as LSL stream `unicorn_eeg`.

### macOS Setup

1. Install liblsl:
   ```bash
   brew install labstreaminglayer/tap/lsl
   ```

2. Both machines must be on the same network.

3. Run the BCI with LSL input:
   ```bash
   uv run python main.py --lsl
   ```

   The `.env` file sets `PYLSL_LIB` automatically. The macOS side will wait up to 10 seconds for the Windows LSL stream to appear.

## Architecture

```
Generator/HybridBlack/LSL (250 Hz, 8ch)
  │
  ▼
Bandpass (1–45 Hz) → Notch50 → Notch60
  │                              │
  ▼                              ▼
TimeSeriesScope              FFT (ws=250, overlap=0.5)
  (debug)                    │           │
                             ▼           ▼
                       SpectrumScope   SSVEPDetector
                        (debug)            │
                                       callback
                                           │
                                           ▼
                                    SSVEPStimulus
                                    [10 Hz]  [15 Hz]
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — wires pipeline, UI, and detector |
| `ssvep_stimulus.py` | Flickering image widget (10 Hz + 15 Hz) |
| `ssvep_detector.py` | FFT power classifier (INode pipeline sink) |
| `images.py` | Placeholder image generator (swap for real images) |
| `lsl_source.py` | LSL receiver source node for cross-machine streaming |
| `windows_stream.py` | Windows-side script: HybridBlack → LSL |
