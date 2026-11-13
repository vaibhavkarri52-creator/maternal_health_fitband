#!/usr/bin/env python3
"""
Unified ESP32 Biosensor Pipeline (with MPU6050 Motion Gate + PPG SQI Gate + 200Hz BP):
Reads ONE 50 Hz CSV (timestamp_ms,ir_ppg,red_ppg,ax,ay,az) and outputs:
  1. Heart Rate (BPM)        - Systolic Peak Detection (SQI-gated)
  2. SpO2 (%)                - Ratio-of-Ratios Quadratic Calibration (SQI-gated)
  3. Blood Pressure (mmHg)   - 4x Upsampled (200 Hz) PPG Pulse Wave Analysis (SQI-gated)
  4. Glucose (mg/dL & State) - Pretrained Linear Regression (model_coeff_v1.csv, SQI-gated)
  5. Sleep Stage (30s epoch) - Pretrained ResUNet (SleepStagePrediction)
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, resample

THIS_DIR = Path(__file__).resolve().parent
MAT_ROOT = THIS_DIR if (THIS_DIR / "strive").exists() else THIS_DIR.parent

GLUCOSE_REPO = MAT_ROOT / "Non-Invasive-Glucose-Monitoring-System"
DEFAULT_COEFF = GLUCOSE_REPO / "models" / "model_coeff_v1.csv"

import sys

SLEEP_REPO = MAT_ROOT / "strive" / "SleepStagePrediction"
# Cross-platform: checks Windows (.venv\Scripts\python.exe), macOS/Linux (.venv/bin/python), or current Python
_win_py = SLEEP_REPO / ".venv" / "Scripts" / "python.exe"
_unix_py = SLEEP_REPO / ".venv" / "bin" / "python"
SLEEP_VENV_PYTHON = _win_py if _win_py.exists() else (_unix_py if _unix_py.exists() else Path(sys.executable))

MPU6050_LSB_PER_G = 16384.0


def bandpass_ppg(sig: np.ndarray, fs: float = 50.0, low: float = 0.5, high: float = 8.0) -> np.ndarray:
    b, a = butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
    return filtfilt(b, a, sig)


def classify_glucose_zone(mg_dl: float, reject_reason: str = "POOR_SIGNAL_SQI") -> str:
    if np.isnan(mg_dl):
        return reject_reason
    if mg_dl < 70.0:
        return "DOWN (<70 mg/dL)"
    if mg_dl > 180.0:
        return "SPIKE (>180 mg/dL)"
    return "NORMAL (70-180 mg/dL)"


def compute_bp_from_ppg(
    ir_ac: np.ndarray,
    fs: float = 50.0,
    upsample_factor: int = 4,
    sbp_offset: float = 0.0,
    dbp_offset: float = 0.0,
):
    """
    Upsamples bandpass-filtered IR PPG by 4x (50 Hz -> 200 Hz) via Fourier resampling
    for 5 ms timing precision, then computes Systolic (SBP) and Diastolic (DBP) BP.
    """
    fs_hi = fs * upsample_factor
    ir_hi = resample(ir_ac, len(ir_ac) * upsample_factor)

    min_dist = int(0.30 * fs_hi)
    prom = 0.35 * np.std(ir_hi)
    peaks, _ = find_peaks(ir_hi, distance=min_dist, prominence=prom)
    valleys, _ = find_peaks(-ir_hi, distance=min_dist, prominence=prom)

    if len(peaks) < 3 or len(valleys) < 3:
        return None

    hr = 60.0 / float(np.mean(np.diff(peaks) / fs_hi))
    widths_samples, _, _, _ = peak_widths(ir_hi, peaks, rel_height=0.5)
    w50_sec = float(np.mean(widths_samples)) / fs_hi

    t_sys_list, t_dia_list = [], []
    for p in peaks:
        prev_v = valleys[valleys < p]
        next_v = valleys[valleys > p]
        if len(prev_v) > 0 and len(next_v) > 0:
            t_s = (p - prev_v[-1]) / fs_hi
            t_d = (next_v[0] - p) / fs_hi
            if 0.06 <= t_s <= 0.45 and 0.15 <= t_d <= 1.20:
                t_sys_list.append(t_s)
                t_dia_list.append(t_d)

    if not t_sys_list:
        return None

    t_sys = float(np.mean(t_sys_list))
    t_dia = float(np.mean(t_dia_list))

    # Empirical PPG Pulse Wave Analysis (PWA) Equations
    sbp = (4.2 / t_sys) + (0.25 * hr) - (18.0 * (t_sys / t_dia)) + 82.0 + sbp_offset
    dbp = (6.5 / t_dia) + (0.18 * hr) - (25.0 * w50_sec) + 58.0 + dbp_offset

    # Physiological bounds & ensure SBP > DBP + 15 mmHg
    sbp = float(np.clip(sbp, 80.0, 200.0))
    dbp = float(np.clip(dbp, 45.0, min(130.0, sbp - 15.0)))

    return sbp, dbp


def analyze_30s_epoch(
    w: pd.DataFrame,
    coeffs: dict,
    fs: float = 50.0,
    sbp_offset: float = 0.0,
    dbp_offset: float = 0.0,
) -> dict:
    """Computes HR, SpO2, BP (SBP/DBP), and Glucose across motionless, SQI-verified 5s windows."""
    sub_n = int(5.0 * fs)
    hr_list, spo2_list, sbp_list, dbp_list, gluc_list = [], [], [], [], []
    motion_passed = 0
    contact_passed = 0

    for start in range(0, len(w) - sub_n + 1, sub_n):
        sub = w.iloc[start : start + sub_n]

        # 1. MPU6050 Motion Gate (>0.08g dynamic acceleration rejected)
        acc_g = sub[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G
        if np.std(np.linalg.norm(acc_g, axis=1)) > 0.08:
            continue
        motion_passed += 1

        # 2. Sensor Contact Gate (Finger must be present on MAX30102)
        ir = sub["ir_ppg"].to_numpy(dtype=np.float64)
        red = sub["red_ppg"].to_numpy(dtype=np.float64)
        ir_dc, red_dc = float(np.mean(ir)), float(np.mean(red))
        if ir_dc < 10000 or red_dc < 10000:
            continue
        contact_passed += 1

        # 3. PPG Signal Quality Index (SQI) Gate
        ir_zero = ir - ir_dc
        total_ac_var = float(np.var(ir_zero))
        if total_ac_var < 1.0:
            continue

        ir_hr = bandpass_ppg(ir, fs=fs, low=0.5, high=4.0)
        red_hr = bandpass_ppg(red, fs=fs, low=0.5, high=4.0)
        cardiac_var = float(np.var(ir_hr))

        sqi_power_ratio = cardiac_var / (total_ac_var + 1e-12)
        if sqi_power_ratio < 0.50:
            continue

        ac_ir = float(np.sqrt(np.mean(ir_hr ** 2)))
        ac_red = float(np.sqrt(np.mean(red_hr ** 2)))
        pi_ir_pct = (ac_ir / ir_dc) * 100.0
        if not (0.02 <= pi_ir_pct <= 15.0):
            continue

        peaks, _ = find_peaks(ir_hr, distance=int(0.30 * fs), prominence=0.35 * np.std(ir_hr))
        if len(peaks) < 3:
            continue

        rr_sec = np.diff(peaks) / fs
        bpm = 60.0 / float(np.mean(rr_sec))
        rr_cv = float(np.std(rr_sec) / (np.mean(rr_sec) + 1e-8))
        if not (35.0 <= bpm <= 220.0) or rr_cv > 0.35:
            continue

        # --- Passed all Motion + Contact + SQI Gates ---
        hr_list.append(bpm)

        # SpO2
        r_ratio = (ac_red / red_dc) / (ac_ir / ir_dc + 1e-12)
        spo2 = np.clip(-45.060 * (r_ratio ** 2) + 30.354 * r_ratio + 94.845, 70.0, 100.0)
        spo2_list.append(float(spo2))

        # Wideband AC (0.5 - 8.0 Hz) for BP Morphology & Glucose
        ir_ac = bandpass_ppg(ir, fs=fs, low=0.5, high=8.0)

        # Blood Pressure (4x upsampled to 200 Hz)
        bp_res = compute_bp_from_ppg(
            ir_ac,
            fs=fs,
            upsample_factor=4,
            sbp_offset=sbp_offset,
            dbp_offset=dbp_offset,
        )
        if bp_res is not None:
            sbp_list.append(bp_res[0])
            dbp_list.append(bp_res[1])

        # Glucose
        ratio = ir_dc / (red_dc + 1e-8)
        variability = float(np.std(ir_ac) / (ir_dc + 1e-8) * 1000.0)
        slope = float(np.mean(np.abs(np.diff(ir_ac))) * fs / (ir_dc + 1e-8) * 1000.0)

        gluc = (
            coeffs["intercept"]
            + coeffs["ratio_coeff"] * ratio
            + coeffs["var_coeff"] * variability
            + coeffs["slope_coeff"] * slope
        )
        gluc_list.append(float(gluc))

    if motion_passed == 0:
        reject_reason = "MOTION_REJECTED"
    elif contact_passed == 0:
        reject_reason = "FINGER_OFF_SENSOR"
    else:
        reject_reason = "POOR_SIGNAL_SQI"

    mean_gluc = round(float(np.mean(gluc_list)), 1) if gluc_list else np.nan
    return {
        "heart_rate_bpm": round(float(np.mean(hr_list)), 1) if hr_list else np.nan,
        "spo2_pct": round(float(np.mean(spo2_list)), 1) if spo2_list else np.nan,
        "sbp_mmhg": round(float(np.mean(sbp_list)), 1) if sbp_list else np.nan,
        "dbp_mmhg": round(float(np.mean(dbp_list)), 1) if dbp_list else np.nan,
        "glucose_mg_dl": mean_gluc,
        "glucose_status": classify_glucose_zone(mean_gluc, reject_reason=reject_reason),
    }


SLEEP_SCRIPT = SLEEP_REPO / "minimal_example" / "run_sleep_pipeline.py"
SLEEP_OUT_CSV = SLEEP_REPO / "minimal_example" / "sleep_predictions.csv"

def run_sleep_subprocess(csv_path: Path, fs: float) -> pd.DataFrame:
    """Runs SleepStagePrediction ResUNet via its dedicated .venv and returns predictions."""
    if not SLEEP_VENV_PYTHON.exists() or not SLEEP_SCRIPT.exists():
        print(f"[WARN] Sleep pipeline not found at {SLEEP_SCRIPT}. Skipping sleep stage inference.")
        return pd.DataFrame()

    cmd = [str(SLEEP_VENV_PYTHON), str(SLEEP_SCRIPT), str(csv_path.resolve()), "--raw-fs", str(int(fs))]
    subprocess.run(cmd, cwd=str(SLEEP_SCRIPT.parent), check=True)
    if SLEEP_OUT_CSV.exists():
        return pd.read_csv(SLEEP_OUT_CSV)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Unified ESP32 Biosensor Pipeline (HR, SpO2, BP, Glucose, Sleep)")
    parser.add_argument("csv_path", help="Path to raw 50 Hz ESP32 CSV file")
    parser.add_argument("--raw-fs", type=float, default=50.0, help="Sampling frequency in Hz (default: 50)")
    parser.add_argument("--sbp-offset", type=float, default=0.0, help="Cuff calibration offset for Systolic BP (mmHg)")
    parser.add_argument("--dbp-offset", type=float, default=0.0, help="Cuff calibration offset for Diastolic BP (mmHg)")
    parser.add_argument("--coeffs", default=str(DEFAULT_COEFF), help="Path to model_coeff_v1.csv")
    parser.add_argument("--out", default="unified_biosensor_report.csv", help="Combined output CSV path")
    parser.add_argument("--stream", action="store_true", help="Stream each evaluated epoch in real-time to the web app")
    parser.add_argument("--stream-url", default="http://localhost:8080/api/telemetry", help="Target webhook URL for live vitals")
    parser.add_argument("--stream-delay", type=float, default=1.0, help="Delay in seconds between streaming epochs")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    df = pd.read_csv(csv_path)
    required = {"ir_ppg", "red_ppg", "ax", "ay", "az"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {list(df.columns)}")

    coeffs = pd.read_csv(args.coeffs).set_index("feature")["coefficient"].to_dict()

    epoch_n = int(30.0 * args.raw_fs)
    rows = []
    for idx, start in enumerate(range(0, len(df) - epoch_n + 1, epoch_n)):
        w = df.iloc[start : start + epoch_n]
        metrics = analyze_30s_epoch(
            w,
            coeffs,
            fs=args.raw_fs,
            sbp_offset=args.sbp_offset,
            dbp_offset=args.dbp_offset,
        )
        rows.append({
            "epoch_idx": idx,
            "start_time_sec": idx * 30,
            **metrics,
        })

    vitals_df = pd.DataFrame(rows)

    sleep_df = run_sleep_subprocess(csv_path, args.raw_fs)
    if not sleep_df.empty and "epoch_idx" in sleep_df.columns:
        merged = pd.merge(
            vitals_df,
            sleep_df[["epoch_idx", "predicted_stage", "prob_wake", "prob_light", "prob_deep", "prob_rem"]],
            on="epoch_idx",
            how="left",
        )
    else:
        merged = vitals_df
        merged["predicted_stage"] = "N/A"

    out_path = Path(args.out).resolve()
    merged.to_csv(out_path, index=False)

    print("\n" + "=" * 84)
    print(f"UNIFIED ESP32 BIOSENSOR SUMMARY ({csv_path.name})")
    print("=" * 84)
    print(f"Total 30s Epochs Evaluated : {len(merged)}")
    print(f"Mean Heart Rate            : {merged['heart_rate_bpm'].mean():.1f} BPM")
    print(f"Mean SpO2                  : {merged['spo2_pct'].mean():.1f} %")
    print(f"Mean Blood Pressure        : {merged['sbp_mmhg'].mean():.1f} / {merged['dbp_mmhg'].mean():.1f} mmHg")
    mean_g = merged["glucose_mg_dl"].mean()
    dominant_reject = merged["glucose_status"].mode().iloc[0] if not merged.empty else "POOR_SIGNAL_SQI"
    print(f"Mean Glucose Estimate      : {mean_g:.1f} mg/dL -> {classify_glucose_zone(mean_g, dominant_reject)}")
    if "predicted_stage" in merged.columns:
        counts = merged["predicted_stage"].value_counts().to_dict()
        print(f"Sleep Stage Distribution   : {counts}")
    print(f"Saved Full Report To       : {out_path}\n")

    display_cols = [
        "epoch_idx",
        "start_time_sec",
        "heart_rate_bpm",
        "spo2_pct",
        "sbp_mmhg",
        "dbp_mmhg",
        "glucose_mg_dl",
        "glucose_status",
        "predicted_stage",
    ]
    print(merged[display_cols].head(10).to_string(index=False))

    if args.stream:
        import urllib.request
        import json
        import time

        print(f"\n[STREAM] Streaming {len(merged)} epochs to {args.stream_url}...")
        for _, row in merged.iterrows():
            payload = {}
            for col in ["heart_rate_bpm", "spo2_pct", "sbp_mmhg", "dbp_mmhg", "glucose_mg_dl", "glucose_status", "predicted_stage", "epoch_idx"]:
                if col in row:
                    val = row[col]
                    if pd.isna(val):
                        payload[col] = None
                    elif isinstance(val, (np.floating, float)):
                        payload[col] = round(float(val), 2)
                    elif isinstance(val, (np.integer, int)):
                        payload[col] = int(val)
                    else:
                        payload[col] = str(val)
            try:
                req = urllib.request.Request(
                    args.stream_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    pass
                hr_str = f"{payload['heart_rate_bpm']} BPM" if payload['heart_rate_bpm'] else "--"
                bp_str = f"{payload['sbp_mmhg']}/{payload['dbp_mmhg']}" if payload['sbp_mmhg'] else "--/--"
                gluc_str = f"{payload['glucose_mg_dl']} mg/dL" if payload['glucose_mg_dl'] else payload.get('glucose_status', '--')
                print(f"  -> Streamed Epoch #{payload.get('epoch_idx')}: HR={hr_str} | BP={bp_str} | Glucose={gluc_str}")
            except Exception as e:
                print(f"  [!] Stream error for epoch #{payload.get('epoch_idx')}: {e}")
            if args.stream_delay > 0:
                time.sleep(args.stream_delay)


if __name__ == "__main__":
    main()