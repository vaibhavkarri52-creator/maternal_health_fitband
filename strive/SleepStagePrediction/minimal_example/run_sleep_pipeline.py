#!/usr/bin/env python3
"""
Cross-Platform (Windows / macOS / Linux) End-to-End Sleep Stage Inference
For ESP32 (MAX30102 + MPU6050) -> MADSOLSEN/SleepStagePrediction

Usage:
  1. Test directly on the repo's sample H5 file:
     python run_sleep_pipeline.py test_subject.h5

  2. Run directly on your ESP32 CSV file (auto-converts CSV -> H5 -> Predictions):
     python run_sleep_pipeline.py dummy_esp32.csv --raw-fs 50
"""

import os
import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample_poly

# ==============================================================================
# 1. OS-Independent Path Resolution (Works on Windows, macOS, and Linux)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "minimal_example" else SCRIPT_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from datasets.dataset_gen import DatasetGenerator
from models import ResUNet

TARGET_FS = 32               # 32 Hz expected by SleepStagePrediction
MPU6050_LSB_PER_G = 16384.0  # +/- 2g scale factor
EVENTS = ["wake", "light", "deep", "rem"]


# ==============================================================================
# 2. Step A: Hardware CSV -> Verified HDF5 (.h5) Converter
# ==============================================================================
def csv_to_h5(csv_path: Path, out_h5_path: Path, raw_fs: int = 50) -> Path:
    df = pd.read_csv(csv_path)
    required_cols = {"ir_ppg", "ax", "ay", "az"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}. Found: {list(df.columns)}")

    ir_raw = df["ir_ppg"].to_numpy(dtype=np.float64)
    acc_raw = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64) / MPU6050_LSB_PER_G

    # Clean MAX30102 IR PPG via NeuroKit2 Elgendi (0.5 - 8 Hz bandpass)
    ppg_cleaned = nk.ppg_clean(ir_raw, sampling_rate=raw_fs, method="elgendi")

    # Rational polyphase resampling from raw_fs (50 Hz) -> TARGET_FS (32 Hz)
    gcd_val = np.gcd(int(TARGET_FS), int(raw_fs))
    up = int(TARGET_FS) // gcd_val
    down = int(raw_fs) // gcd_val

    ppg_32hz = resample_poly(ppg_cleaned, up, down).astype(np.float64)
    acc_32hz = resample_poly(acc_raw, up, down, axis=0).astype(np.float64)

    # Trim to an exact multiple of 30-second epochs (30s * 32Hz = 960 samples)
    epoch_samples = 30 * TARGET_FS
    usable_samples = (len(ppg_32hz) // epoch_samples) * epoch_samples
    if usable_samples == 0:
        raise ValueError("Recording must be at least 30 seconds long.")

    ppg_out = ppg_32hz[:usable_samples, np.newaxis]  # (N, 1) float64
    acc_out = acc_32hz[:usable_samples, :]           # (N, 3) float64
    total_sec = int(usable_samples // TARGET_FS)

    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5_path, "w") as h5:
        ds_acc = h5.create_dataset("acc_signal", data=acc_out, dtype=np.float64)
        ds_acc.attrs["fs"] = TARGET_FS

        ds_ppg = h5.create_dataset("ppg_signal", data=ppg_out, dtype=np.float64)
        ds_ppg.attrs["fs"] = TARGET_FS

        # Placeholder int64 stage groups so DatasetGenerator masks/loads cleanly
        for i, stage in enumerate(EVENTS):
            grp = h5.create_group(stage)
            start_sec = min(i * 30, max(0, total_sec - 30))
            grp.create_dataset("start", data=np.array([start_sec], dtype=np.int64))
            grp.create_dataset("duration", data=np.array([30], dtype=np.int64))

    print(f"[OK] Converted CSV -> H5: {out_h5_path}")
    print(f"     acc_signal: {acc_out.shape}, ppg_signal: {ppg_out.shape} ({total_sec // 60}m {total_sec % 60}s)")
    return out_h5_path


# ==============================================================================
# 3. Step B: Pretrained ResUNet Inference (Exact Notebook Configuration)
# ==============================================================================
def run_inference(h5_file_path: Path, weights_path: Path, out_csv_path: Path):
    signals_format = {
        "ACC_merge_fft_spec": {
            "add": True,
            "fs_post": 0.5,
            "h5_path": "acc_signal",
            "dimensions": [32, 1],
            "channel_idx": [0, 1, 2],
            "preprocessing": [
                {"args": {"window_size": 30}, "type": "median"},
                {"args": {"iqr_window": 300, "median_window": 300}, "type": "iqr_normalization_adaptive"},
                {"args": {"threshold": 20}, "type": "clip_by_iqr"},
                {"args": {"nfft": 512, "f_max": 6, "f_min": 0, "f_sub": 3, "window": 320, "noverlap": 256}, "type": "cal_psd"},
            ],
            "transformations": {},
            "batch_normalization": {},
        },
        "PPG_fft_spec": {
            "add": False,
            "fs_post": 0.5,
            "h5_path": "ppg_signal",
            "dimensions": [32, 1],
            "channel_idx": [0],
            "preprocessing": [
                {"args": {}, "type": "zscore"},
                {"args": {}, "type": "change_PPG_direction"},
                {"args": {"iqr_window": 301, "median_window": 301}, "type": "iqr_normalization_adaptive"},
                {"args": {"threshold": 20}, "type": "clip_by_iqr"},
                {"args": {"nfft": 512, "f_max": 2.1, "f_min": 0.1, "f_sub": 1, "window": 320, "noverlap": 256}, "type": "cal_psd"},
            ],
            "transformations": {},
            "batch_normalization": {},
        },
    }

    events_format = [
        {"name": ev, "h5_path": ev, "probability": 1.0 / len(EVENTS)}
        for ev in EVENTS
    ]

    dataset_params = {
        "h5_directory": str(h5_file_path.parent),
        "signals_format": signals_format,
        "window": 30 * (2 ** 10),
        "number_of_channels": len(signals_format),
        "events_format": events_format,
        "prediction_resolution": 30,
        "overlap": 0.25,
        "minimum_overlap": 0.1,
        "batch_size": 1,
        "cache_data": False,
        "n_jobs": 1,
        "use_mask": False,
        "load_signal_in_RAM": True,
    }

    ds_test = DatasetGenerator(
        records=[h5_file_path.name],
        mode="inference",
        **dataset_params
    )

    model_params = {
        "input_shape": [int(ds_test.fsTime * ds_test.window), ds_test.nSpace, ds_test.nChannels],
        "num_classes": len(EVENTS),
        "num_outputs": ds_test.window // ds_test.prediction_resolution,
        "depth": 9,
        "init_filter_num": 16,
        "filter_increment_factor": 2 ** (1 / 3),
        "max_pool_size": (2, 2),
        "kernel_size": (16, 3),
    }

    print(f"[INFO] Building ResUNet and loading weights: {weights_path}")
    model = ResUNet(**model_params)
    model.load_weights(filepath=str(weights_path))

    with h5py.File(h5_file_path, "r") as h5:
        total_samples = h5["ppg_signal"].shape[0]
    actual_epochs = total_samples // (30 * TARGET_FS)

    all_probs = []
    for idx in range(len(ds_test)):
        sig, _ = ds_test[idx]
        preds = model.predict(sig, verbose=0)
        for b in range(preds.shape[0]):
            all_probs.append(preds[b])

    probs_concat = np.concatenate(all_probs, axis=0)[:actual_epochs]
    pred_indices = np.argmax(probs_concat, axis=1)
    pred_labels = [EVENTS[i] for i in pred_indices]

    results_df = pd.DataFrame({
        "epoch_idx": np.arange(len(pred_labels)),
        "start_time_sec": np.arange(len(pred_labels)) * 30,
        "predicted_stage": pred_labels,
        "prob_wake": np.round(probs_concat[:, 0], 4),
        "prob_light": np.round(probs_concat[:, 1], 4),
        "prob_deep": np.round(probs_concat[:, 2], 4),
        "prob_rem": np.round(probs_concat[:, 3], 4),
    })
    results_df.to_csv(out_csv_path, index=False)

    print(f"\n[OK] Sleep Stage Predictions Saved -> {out_csv_path}")
    print("Stage Summary (30s epochs):")
    print(results_df["predicted_stage"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-platform Sleep Stage Prediction Pipeline")
    parser.add_argument("input_file", help="Path to raw ESP32 .csv OR existing .h5 file")
    parser.add_argument("--raw-fs", type=int, default=50, help="ESP32 CSV sampling rate in Hz (default: 50)")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models" / "pretrained_weights" / "model-best.h5"),
        help="Path to pretrained .h5 model weights",
    )
    parser.add_argument("--out-csv", default=str(SCRIPT_DIR / "sleep_predictions.csv"), help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        input_path = (SCRIPT_DIR / args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if input_path.suffix.lower() == ".csv":
        h5_dir = REPO_ROOT / "data" / "h5"
        h5_path = csv_to_h5(input_path, h5_dir / f"{input_path.stem}.h5", raw_fs=args.raw_fs)
    elif input_path.suffix.lower() in {".h5", ".hdf5"}:
        h5_path = input_path
    else:
        raise ValueError("Input file must be a .csv or .h5 file")

    run_inference(h5_path, Path(args.weights).resolve(), Path(args.out_csv).resolve())