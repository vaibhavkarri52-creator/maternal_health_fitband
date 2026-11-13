#!/usr/bin/env python3
"""
Bridges raw ESP32 CSV logs (timestamp_ms,ir_ppg,red_ppg,ax,ay,az) to
Non-Invasive-Glucose-Monitoring-System (models/model_coeff_v2.csv + train_model.py).

Usage:
  1. Predict immediately using repo's model_coeff_v2.csv (outputs Down / Normal / Spike):
     python software/esp32_glucose_bridge.py predict path/to/esp32.csv

  2. Export extracted features in train_model.py's exact schema (for personal calibration):
     python software/esp32_glucose_bridge.py extract path/to/esp32.csv --glucose 105.0 --out data/training/cal_01.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COEFF_V2 = REPO_ROOT / "models" / "model_coeff_v2.csv"
MPU6050_LSB_PER_G = 16384.0


def bandpass_ppg(sig: np.ndarray, fs: float = 50.0, low: float = 0.5, high: float = 8.0) -> np.ndarray:
    b, a = butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
    return filtfilt(b, a, sig)


def extract_repo_features(csv_path: Path, fs: float = 50.0, win_sec: int = 5) -> pd.DataFrame:
    """Computes motion-gated (ratio, variability, slope, pulse_rate, acdc_ratio) per 5s window."""
    df = pd.read_csv(csv_path)
    required = {"ir_ppg", "red_ppg", "ax", "ay", "az"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required}. Found: {list(df.columns)}")

    win_n = int(win_sec * fs)
    rows = []

    for start in range(0, len(df) - win_n + 1, win_n):
        w = df.iloc[start : start + win_n]

        # 1. MPU6050 Motion Gate (reject window if dynamic movement > 0.08g)
        acc_g = w[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G
        if np.std(np.linalg.norm(acc_g, axis=1)) > 0.08:
            continue

        ir = w["ir_ppg"].to_numpy(dtype=np.float64)
        red = w["red_ppg"].to_numpy(dtype=np.float64)
        if np.mean(ir) < 10000 or np.mean(red) < 10000:
            continue

        # 2. Conditioned AC/DC & Repo Features (matches model_coeff_v2.csv & train_model.py)
        ir_ac = bandpass_ppg(ir, fs=fs)
        red_ac = bandpass_ppg(red, fs=fs)

        ir_dc, red_dc = np.mean(ir), np.mean(red)
        ratio = ir_dc / (red_dc + 1e-8)                       # IR/Red optical ratio
        variability = float(np.std(ir_ac) / (ir_dc + 1e-8) * 1000.0)  # Normalized AC variability
        slope = float(np.mean(np.abs(np.diff(ir_ac))) * fs / (ir_dc + 1e-8) * 1000.0)

        # Derived features expected by model_coeff_v2.csv & train_model.py
        pulse_rate = 60.0 / (variability + 1e-6)
        acdc_ratio = variability / (ratio + 1e-8)

        ts_ms = int(w["timestamp_ms"].iloc[0]) if "timestamp_ms" in w.columns else int(start / fs * 1000)
        rows.append({
            "timestamp": ts_ms,
            "ratio": round(ratio, 4),
            "variability": round(variability, 4),
            "slope": round(slope, 4),
            "pulse_rate": round(pulse_rate, 4),
            "acdc_ratio": round(acdc_ratio, 4),
        })

    return pd.DataFrame(rows)


def classify_glucose_zone(mg_dl: float) -> str:
    if mg_dl < 70.0:
        return "DOWN (Hypo <70 mg/dL)"
    if mg_dl > 180.0:
        return "SPIKE (Hyper >180 mg/dL)"
    return "NORMAL (70-180 mg/dL)"


def run_predict(csv_path: Path, coeff_path: Path, fs: float):
    feat_df = extract_repo_features(csv_path, fs=fs)
    if feat_df.empty:
        print("[WARN] No valid motionless windows found (hold finger steady for >=5s).")
        return

    coeffs = pd.read_csv(coeff_path).set_index("feature")["coefficient"].to_dict()

    # Exact linear equation from models/model_coeff_v2.csv
    pred_mg_dl = (
        coeffs["intercept"]
        + coeffs["ratio_coeff"] * feat_df["ratio"]
        + coeffs["var_coeff"] * feat_df["variability"]
        + coeffs["slope_coeff"] * feat_df["slope"]
        + coeffs.get("hrv_coeff", 0.0) * feat_df["pulse_rate"]
        + coeffs.get("acdc_coeff", 0.0) * feat_df["acdc_ratio"]
    )
    feat_df["pred_glucose_mg_dl"] = pred_mg_dl.round(1)
    feat_df["status"] = feat_df["pred_glucose_mg_dl"].apply(classify_glucose_zone)

    mean_val = float(feat_df["pred_glucose_mg_dl"].mean())
    print(f"\n=== Glucose Classification ({csv_path.name}) ===")
    print(f"Motionless 5s Windows : {len(feat_df)}")
    print(f"Estimated Mean        : {mean_val:.1f} mg/dL")
    print(f"Glycemic Status       : {classify_glucose_zone(mean_val)}\n")
    print(feat_df[["timestamp", "ratio", "variability", "slope", "pred_glucose_mg_dl", "status"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 Bridge for Non-Invasive-Glucose-Monitoring-System")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Classify Down / Normal / Spike using model_coeff_v2.csv")
    p_pred.add_argument("csv_path", help="Raw ESP32 CSV path")
    p_pred.add_argument("--coeffs", default=str(DEFAULT_COEFF_V2), help="Path to model_coeff_v2.csv")
    p_pred.add_argument("--raw-fs", type=float, default=50.0, help="Sampling rate in Hz")

    p_ext = sub.add_parser("extract", help="Convert raw ESP32 CSV into train_model.py format")
    p_ext.add_argument("csv_path", help="Raw ESP32 CSV path")
    p_ext.add_argument("--glucose", type=float, required=True, help="Reference glucometer value (mg/dL)")
    p_ext.add_argument("--out", required=True, help="Output CSV path for train_model.py")
    p_ext.add_argument("--raw-fs", type=float, default=50.0, help="Sampling rate in Hz")

    args = parser.parse_args()
    if args.cmd == "predict":
        run_predict(Path(args.csv_path), Path(args.coeffs), args.raw_fs)
    elif args.cmd == "extract":
        df_out = extract_repo_features(Path(args.csv_path), fs=args.raw_fs)
        df_out["glucose"] = args.glucose
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out, index=False)
        print(f"[OK] Saved {len(df_out)} calibrated windows -> {args.out}")
        