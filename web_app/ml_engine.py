import math
from typing import Any, Dict, Optional


def _clean(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


class MaternalMLEngine:
    def __init__(self, models_dir: Optional[str] = None):
        self.models = {}
        print("[Wrapper] Initialized zero-model mode (consuming run_all_biosensors.py output directly).")

    def classify(self, raw_data: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        hr = _clean(raw_data.get("heart_rate_bpm") if "heart_rate_bpm" in raw_data else raw_data.get("hr"))
        spo2 = _clean(raw_data.get("spo2_pct") if "spo2_pct" in raw_data else raw_data.get("spo2"))
        sbp = _clean(raw_data.get("sbp_mmhg") if "sbp_mmhg" in raw_data else raw_data.get("sbp"))
        dbp = _clean(raw_data.get("dbp_mmhg") if "dbp_mmhg" in raw_data else raw_data.get("dbp"))
        glucose = _clean(raw_data.get("glucose_mg_dl") if "glucose_mg_dl" in raw_data else raw_data.get("glucose"))

        glucose_status = str(raw_data.get("glucose_status") or "Awaiting Sensor Input")
        stage = str(raw_data.get("predicted_stage") or "unknown").lower()

        stage_map = {
            "wake": (False, "Awake / Upright", "Active / Awake"),
            "light": (True, "Light Sleep (Lateral)", "Sleeping (Light)"),
            "deep": (True, "Deep Sleep (Lateral)", "Sleeping (Deep)"),
            "rem": (True, "REM Sleep (Lateral)", "Sleeping (REM)"),
        }
        is_recumbent, posture_label, activity_label = stage_map.get(stage, (False, "Resting", "Resting"))

        hr_r = round(hr, 1) if hr is not None else None
        spo2_r = round(spo2, 1) if spo2 is not None else None
        sbp_r = round(sbp, 1) if sbp is not None else None
        dbp_r = round(dbp, 1) if dbp is not None else None
        gluc_r = round(glucose, 1) if glucose is not None else None

        vitals = {
            "heart_rate_bpm": hr_r, "heart_rate": hr_r, "hr": hr_r,
            "spo2_pct": spo2_r, "spo2": spo2_r,
            "sbp_mmhg": sbp_r, "bp_systolic": sbp_r, "sbp": sbp_r,
            "dbp_mmhg": dbp_r, "bp_diastolic": dbp_r, "dbp": dbp_r,
            "glucose_mg_dl": gluc_r, "glucose": gluc_r,
            "glucose_status": glucose_status,
            "sleep_stage": stage,
        }

        kinematics = {
            "vm_mean_g": 1.0,
            "vm_std_g": 0.05,
            "is_recumbent": is_recumbent,
            "supine_alert": False,
            "activity_code": 0 if is_recumbent else 1,
            "posture_label": posture_label,
            "activity_label": activity_label,
        }

        return {
            "epoch_idx": raw_data.get("epoch_idx"),
            "start_time_sec": raw_data.get("start_time_sec"),
            "vitals": vitals,
            "kinematics": kinematics,
            "predicted_stage": stage,
            "clinical_status": glucose_status,
        }

    infer = classify
    predict_vitals = classify


ml_engine = MaternalMLEngine()