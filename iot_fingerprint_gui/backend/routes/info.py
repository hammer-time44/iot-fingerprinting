"""
Information and evaluation API routes.
"""
import os
import json
from fastapi import APIRouter

router = APIRouter()

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_DIR = os.path.dirname(BACKEND_DIR)
EVAL_PATH = os.path.join(GUI_DIR, "assets", "evaluation", "evaluation_data.json")
META_PATH = os.path.join(GUI_DIR, "assets", "metadata.json")


@router.get("/model/info")
def get_model_info():
    """Return model configuration, features, trained classes, and calibrated metrics."""
    with open(EVAL_PATH, "r") as f:
        eval_data = json.load(f)
    with open(META_PATH, "r") as f:
        meta_data = json.load(f)

    return {
        "model_type": meta_data.get("model_type", "RandomForestClassifier"),
        "n_estimators": meta_data.get("n_estimators", 100),
        "window_size": meta_data.get("window_size", 128),
        "feature_type": meta_data.get("feature_type", "13 Statistical Timing Moments (log1p IAT)"),
        "trained_classes": meta_data.get("training_devices", []),
        "threshold": eval_data.get("threshold", 0.8400),
        "calibration_method": eval_data.get("calibration_method", ""),
        "known_test_accuracy": eval_data.get("known_test_accuracy", 0.0),
        "known_test_false_alarm_rate": eval_data.get("known_test_false_alarm_rate", 0.0),
        "overall_unknown_detection_rate": eval_data.get("overall_unknown_detection_rate", 0.0),
        "confusion_matrix": eval_data.get("confusion_matrix", {}),
        "feature_importances": eval_data.get("feature_importances", [])
    }


@router.get("/evaluation")
def get_evaluation_matrix():
    """Return per-device evaluation for all 27 devices in the UNSW benchmark."""
    with open(EVAL_PATH, "r") as f:
        eval_data = json.load(f)
    return eval_data
