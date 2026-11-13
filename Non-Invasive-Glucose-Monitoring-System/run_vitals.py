#!/usr/bin/env python3
"""
Formula-Based Heart Rate (BPM) & SpO2 (%) Calculator for ESP32 CSV Logs
Input: timestamp_ms,ir_ppg,red_ppg,ax,ay,az at 50 Hz
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

MPU6050_LSB_PER_G = 16384.0


def bandpass_ppg(sig: np.ndarray, fs: float = 50.0, low: float = 0.5, high: float = 4.0) -> np.ndarray:
    b, a = butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
    return filtfilt(b, a, sig)


def compute_window_vitals(ir: np.ndarray, red: np.ndarray, fs: float = 50.0):
    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))
    if ir_dc < 10000 or red_dc < 10000:
        return None  # Finger not placed on sensor

    ir_ac_sig = bandpass_ppg(ir, fs=fs)
    red_ac_sig = bandpass_ppg(red, fs=fs)

    # 1. Heart Rate (BPM) from IR systolic peaks (min distance 0.30s -> max 200 BPM)
    min_dist = int(0.30 * fs)
    prominence = 0.3 * np.std(ir_ac_sig)
    peaks, _ = find_peaks(ir_ac_sig, distance=min_dist, prominence=prominence)

    if len(peaks) < 2:
        return None

    rr_intervals_sec = np.diff(peaks) / fs
    bpm = 60.0 / float(np.mean(rr_intervals_sec))
    if not (35.0 <= bpm <= 220.0):
        return None

    # 2. SpO2 (%) via Ratio of Ratios (R) + Maxim quadratic calibration
    ac_ir = float(np.sqrt(np.mean(ir_ac_sig ** 2)))
    ac_red = float(np.sqrt(np.mean(red_ac_sig ** 2)))

    r_ratio = (ac_red / red_dc) / (ac_ir / ir_dc + 1e-12)
    spo2 = -45.060 * (r_ratio ** 2) + 30.354 * r_ratio + 94.845
    spo2 = float(np.clip(spo2, 70.0, 100.0))

    return round(bpm, 1), round(spo2, 1), round(r_ratio, 4)


def run_vitals(csv_path: Path, fs: float = 50.0, win_sec: int = 10):
    df = pd.read_csv(csv_path)
    required = {"ir_ppg", "red_ppg", "ax", "ay", "az"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}. Found: {list(df.columns)}")

    win_n = int(win_sec * fs)
    rows = []

    for start in range(0, len(df) - win_n + 1, win_n):
        w = df.iloc[start : start + win_n]

        # MPU6050 Motion Gate (>0.08g dynamic movement rejected)
        acc_g = w[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G
        if np.std(np.linalg.norm(acc_g, axis=1)) > 0.08:
            continue

        res = compute_window_vitals(
            w["ir_ppg"].to_numpy(dtype=np.float64),
            w["red_ppg"].to_numpy(dtype=np.float64),
            fs=fs,
        )
        if res is None:
            continue

        bpm, spo2, r_ratio = res
        ts_ms = int(w["timestamp_ms"].iloc[0]) if "timestamp_ms" in w.columns else int(start / fs * 1000)
        rows.append({"timestamp_ms": ts_ms, "heart_rate_bpm": bpm, "spo2_pct": spo2, "R_ratio": r_ratio})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("[WARN] No valid motionless windows found.")
        return

    print(f"\n=== Vitals Summary ({csv_path.name}) ===")
    print(f"Valid Motionless Windows : {len(out_df)}")
    print(f"Mean Heart Rate          : {out_df['heart_rate_bpm'].mean():.1f} BPM")
    print(f"Mean SpO2                : {out_df['spo2_pct'].mean():.1f} %\n")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formula-Based HR & SpO2 Calculator")
    parser.add_argument("csv_path", help="Path to raw 50 Hz ESP32 CSV file")
    parser.add_argument("--raw-fs", type=float, default=50.0, help="Sampling rate in Hz")
    parser.add_argument("--win-sec", type=int, default=10, help="Window length in seconds")
    args = parser.parse_args()

    run_vitals(Path(args.csv_path), fs=args.raw_fs, win_sec=args.win_sec)