#!/usr/bin/env python3
"""
Single-File Pure-Inference Glucose Pipeline (ESP32 CSV -> Down / Normal / Spike)
Uses unscaled pre-trained coefficients (models/model_coeff_v1.csv).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COEFF = SCRIPT_DIR / "models" / "model_coeff_v1.csv"
MPU6050_LSB_PER_G = 16384.0


def bandpass_ppg(sig: np.ndarray, fs: float = 50.0, low: float = 0.5, high: float = 8.0) -> np.ndarray:
    b, a = butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
    return filtfilt(b, a, sig)


def classify_zone(mg_dl: float) -> str:
    if mg_dl < 70.0:
        return "DOWN (Hypo <70 mg/dL)"
    if mg_dl > 180.0:
        return "SPIKE (Hyper >180 mg/dL)"
    return "NORMAL (70-180 mg/dL)"


def extract_features(df: pd.DataFrame, fs: float = 50.0, win_sec: int = 5) -> pd.DataFrame:
    # Accept pre-extracted feature CSVs (like prediction_sample.csv) directly
    if {"ratio", "variability", "slope"}.issubset(df.columns):
        out = df.copy()
        if "timestamp_ms" not in out.columns and "timestamp" in out.columns:
            out["timestamp_ms"] = out["timestamp"]
        return out

    required = {"ir_ppg", "red_ppg", "ax", "ay", "az"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required} or pre-extracted features. Found: {list(df.columns)}")

    win_n = int(win_sec * fs)
    rows = []

    for start in range(0, len(df) - win_n + 1, win_n):
        w = df.iloc[start : start + win_n]

        # MPU6050 Motion Gate (>0.08g dynamic movement rejected)
        acc_g = w[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G
        if np.std(np.linalg.norm(acc_g, axis=1)) > 0.08:
            continue

        ir = w["ir_ppg"].to_numpy(dtype=np.float64)
        red = w["red_ppg"].to_numpy(dtype=np.float64)
        if np.mean(ir) < 10000 or np.mean(red) < 10000:
            continue

        ir_ac = bandpass_ppg(ir, fs=fs)
        ir_dc, red_dc = np.mean(ir), np.mean(red)

        ratio = ir_dc / (red_dc + 1e-8)
        variability = float(np.std(ir_ac) / (ir_dc + 1e-8) * 1000.0)
        slope = float(np.mean(np.abs(np.diff(ir_ac))) * fs / (ir_dc + 1e-8) * 1000.0)

        ts_ms = int(w["timestamp_ms"].iloc[0]) if "timestamp_ms" in w.columns else int(start / fs * 1000)
        rows.append({
            "timestamp_ms": ts_ms,
            "ratio": ratio,
            "variability": variability,
            "slope": slope,
        })

    return pd.DataFrame(rows)


def run_prediction(csv_path: Path, coeff_path: Path, fs: float = 50.0, win_sec: int = 5):
    df = pd.read_csv(csv_path)
    feat_df = extract_features(df, fs=fs, win_sec=win_sec)
    if feat_df.empty:
        print("[WARN] No valid motionless windows found.")
        return

    coeffs = pd.read_csv(coeff_path).set_index("feature")["coefficient"].to_dict()

    pred_mg_dl = (
        coeffs["intercept"]
        + coeffs["ratio_coeff"] * feat_df["ratio"]
        + coeffs["var_coeff"] * feat_df["variability"]
        + coeffs["slope_coeff"] * feat_df["slope"]
    )

    feat_df["pred_glucose_mg_dl"] = pred_mg_dl.round(1)
    feat_df["status"] = feat_df["pred_glucose_mg_dl"].apply(classify_zone)

    mean_val = float(feat_df["pred_glucose_mg_dl"].mean())
    print(f"\n=== Pure-Inference Glucose Result ({csv_path.name}) ===")
    print(f"Motionless Windows Evaluated : {len(feat_df)}")
    print(f"Estimated Mean Glucose       : {mean_val:.1f} mg/dL")
    print(f"Overall Glycemic State       : {classify_zone(mean_val)}\n")
    print(feat_df[["timestamp_ms", "ratio", "variability", "slope", "pred_glucose_mg_dl", "status"]].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure-Inference ESP32 Glucose Pipeline")
    parser.add_argument("csv_path", help="Path to raw ESP32 CSV log or feature CSV")
    parser.add_argument("--coeffs", default=str(DEFAULT_COEFF), help="Path to coefficient CSV file")
    parser.add_argument("--raw-fs", type=float, default=50.0, help="Sampling rate in Hz")
    args = parser.parse_args()

    run_prediction(Path(args.csv_path), Path(args.coeffs), fs=args.raw_fs)