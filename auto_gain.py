#!/usr/bin/env python3

import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import welch
import time

# =========================
# CONFIG
# =========================

CENTER_FREQ = 436.500e6
SAMPLE_RATE = 2.4e6
NUM_SAMPLES = 256 * 1024

GAINS = [
    0.0,
    9.9,
    14.4,
    19.7,
    24.7,
    28.0,
    29.7,
    32.8,
    36.4,
    37.2,
    38.6,
    40.2,
    42.1,
    43.4,
    44.5,
    48.0,
    49.6
]

print("Opening RTL-SDR...")

sdr = RtlSdr()

sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ

best_gain = None
best_score = -999

print("\nScanning gains...\n")

for gain in GAINS:

    sdr.gain = gain

    time.sleep(1)

    samples = sdr.read_samples(NUM_SAMPLES)

    # IQ clipping detection
    max_iq = np.max(np.abs(samples))

    if max_iq > 0.95:
        print(f"Gain {gain:5.1f} dB -> OVERLOAD")
        continue

    # Power Spectral Density
    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE,
        nperseg=2048,
        return_onesided=False
    )

    psd_db = 10 * np.log10(psd + 1e-12)

    signal_peak = np.max(psd_db)
    noise_floor = np.median(psd_db)

    snr = signal_peak - noise_floor

    # Score system
    score = snr - (noise_floor * 0.05)

    print(
        f"Gain {gain:5.1f} dB | "
        f"Peak {signal_peak:6.1f} | "
        f"Noise {noise_floor:6.1f} | "
        f"SNR {snr:5.1f}"
    )

    if score > best_score:
        best_score = score
        best_gain = gain

sdr.close()

print("\n======================")
print(f"BEST GAIN : {best_gain} dB")
print("======================")