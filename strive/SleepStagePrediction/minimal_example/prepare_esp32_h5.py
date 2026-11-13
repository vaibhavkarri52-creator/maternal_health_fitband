#!/usr/bin/env python3
"""
Exact Hardware-to-HDF5 Bridge for MADSOLSEN/SleepStagePrediction
Verified against minimal_example/test_subject.h5:
  - acc_signal : (N, 3) float64
  - ppg_signal : (N, 1) float64
  - deep/start, deep/duration   : (K,) int64
  - light/start, light/duration : (K,) int64
  - rem/start, rem/duration     : (K,) int64
  - wake/start, wake/duration   : (K,) int64
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample_poly

TARGET_FS = 32               # 32 Hz expected by SleepStagePrediction
MPU6050_LSB_PER_G = 16384.0  # +/- 2g scale factor


def build_h5_for_sleep_stage_prediction(
    csv_path: str,
    output_h5_path: str = "../data/h5/subject_01.h5",
    raw_fs: int = 50,
):
    df = pd.read_csv(csv_path)
    required_cols = {"ir_ppg", "ax", "ay", "az"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}. Found: {list(df.columns)}")

    ir_raw = df["ir_ppg"].to_numpy(dtype=np.float64)
    acc_raw = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G

    # 1. Clean raw MAX30102 IR PPG via NeuroKit2 Elgendi (0.5 - 8 Hz bandpass)
    ppg_cleaned = nk.ppg_clean(ir_raw, sampling_rate=raw_fs, method="elgendi")

    # 2. Polyphase rational resampling from raw_fs (50 Hz) -> TARGET_FS (32 Hz)
    gcd_val = np.gcd(int(TARGET_FS), int(raw_fs))
    up = int(TARGET_FS) // gcd_val
    down = int(raw_fs) // gcd_val

    ppg_32hz = resample_poly(ppg_cleaned, up, down).astype(np.float64)
    acc_32hz = resample_poly(acc_raw, up, down, axis=0).astype(np.float64)

    # 3. Trim to an exact multiple of 30-second epochs (30s * 32Hz = 960 samples)
    epoch_samples = 30 * TARGET_FS
    usable_samples = (len(ppg_32hz) // epoch_samples) * epoch_samples
    if usable_samples == 0:
        raise ValueError("Recording must be at least 30 seconds long.")

    ppg_out = ppg_32hz[:usable_samples, np.newaxis]  # (N, 1) float64
    acc_out = acc_32hz[:usable_samples, :]           # (N, 3) float64
    total_sec = int(usable_samples // TARGET_FS)

    # 4. Write HDF5 matching test_subject.h5 schema (float64 signals, int64 groups)
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
    with h5py.File(output_h5_path, "w") as h5:
        ds_acc = h5.create_dataset("acc_signal", data=acc_out, dtype=np.float64)
        ds_acc.attrs["fs"] = TARGET_FS

        ds_ppg = h5.create_dataset("ppg_signal", data=ppg_out, dtype=np.float64)
        ds_ppg.attrs["fs"] = TARGET_FS

        # Dummy int64 stage events so BalancedDataset indexing never hits empty-array errors
        # Splits total duration across the 4 groups as valid 30s int64 intervals
        stage_names = ["wake", "light", "deep", "rem"]
        for i, stage in enumerate(stage_names):
            grp = h5.create_group(stage)
            start_sec = min(i * 30, max(0, total_sec - 30))
            grp.create_dataset("start", data=np.array([start_sec], dtype=np.int64))
            grp.create_dataset("duration", data=np.array([30], dtype=np.int64))

    print(f"[OK] Written: {output_h5_path}")
    print(f"     acc_signal : {acc_out.shape} {acc_out.dtype}")
    print(f"     ppg_signal : {ppg_out.shape} {ppg_out.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to ESP32 CSV log")
    parser.add_argument("--out", default="../data/h5/subject_01.h5", help="Output .h5 path")
    parser.add_argument("--raw-fs", type=int, default=50, help="ESP32 sampling rate (Hz)")
    args = parser.parse_args()

    build_h5_for_sleep_stage_prediction(args.csv_path, args.out, args.raw_fs)