
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


def load_calibration_parameters(path: str | Path) -> dict:
    parameter_path = Path(path)
    if not parameter_path.is_file():
        raise FileNotFoundError(f"PathoMENU calibration parameter file does not exist: {parameter_path}")
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    required = {"params", "monotonic_pchip"}
    missing = required.difference(parameters)
    if missing:
        raise ValueError(f"Calibration parameters are missing fields: {sorted(missing)}")
    return parameters


def apply_pathomenu_calibration(raw_scores, calibration_parameters: dict) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    params = calibration_parameters["params"]
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    transformed = 1.0 / (1.0 + np.exp(-(float(params["a"]) * logits + float(params["b"]))))

    base_scores = np.empty_like(transformed)
    left_mask = transformed < 0.5
    left_u = 2.0 * transformed[left_mask]
    left_exponent = float(params["gamma_left"]) + float(params["tail_left"]) * (1.0 - left_u) ** 2
    base_scores[left_mask] = 0.5 * left_u ** left_exponent

    right_v = 2.0 * (1.0 - transformed[~left_mask])
    right_exponent = float(params["gamma_right"]) + float(params["tail_right"]) * (1.0 - right_v) ** 2
    base_scores[~left_mask] = 1.0 - 0.5 * right_v ** right_exponent

    pchip = calibration_parameters["monotonic_pchip"]
    interpolator = PchipInterpolator(
        np.asarray(pchip["x_knots"], dtype=np.float64),
        np.asarray(pchip["y_knots"], dtype=np.float64),
        extrapolate=False,
    )
    calibrated = interpolator(np.clip(base_scores, 0.0, 1.0))
    return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)
