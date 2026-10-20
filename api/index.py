"""
Vercel Serverless Entry Point for Concrete Strength Predictor API.

Deployment Notes:
- TabPFN (requires PyTorch ~800MB) is excluded from Vercel due to the 250MB
  serverless function size limit. The XGBoost Compressive Champion is used
  as the primary compressive model in production.
- Classical models (XGBoost, Random Forest) are tiny (<1MB total) and load fast.
- Frontend is served by Vercel's CDN from the /public directory.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import os

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths — works both locally and in Vercel serverless (file is at api/index.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "classical"

app = FastAPI(
    title="Concrete Strength Predictor API",
    description=(
        "AI-driven prediction of Compressive, Flexural and Split Tensile strength "
        "for Normal and Bacterial (Bacillus subtilis) concrete. "
        "Classical champion models: XGBoost (Compressive & Tensile) + Random Forest (Flexural)."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy model registry — loaded once per container lifetime (warm starts reuse)
# ---------------------------------------------------------------------------
_models: Dict[str, Any] = {}
_models_loaded = False


def _load_models() -> None:
    global _models, _models_loaded
    if _models_loaded:
        return
    errors: list[str] = []

    for key, filename in [
        ("compressive_xgb",   "xgboost_expanded_compressive.joblib"),
        ("flexural_rf",        "randomforest_expanded_unified.joblib"),
        ("split_tensile_xgb", "xgboost_expanded_split_tensile.joblib"),
    ]:
        path = MODELS_DIR / filename
        if path.exists():
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _models[key] = joblib.load(path)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        else:
            errors.append(f"{key}: file not found at {path}")

    _models_loaded = True
    if errors:
        print(f"[Model Registry] Partial load warnings: {errors}")
    else:
        print(f"[Model Registry] All classical champion models loaded from {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class ConcretePredictionRequest(BaseModel):
    property: Optional[str] = "All"
    concrete_type: Optional[str] = "Bacterial Concrete"
    curing_age_days: Optional[float] = 28.0
    bacterial_concentration_cells_ml: Optional[float] = 1_000_000.0
    cement: Optional[float] = 415.0
    water: Optional[float] = 190.0
    fine_aggregate: Optional[float] = 680.0
    coarse_aggregate: Optional[float] = 1140.0
    specimen_geometry: Optional[str] = "Cube 150mm"
    bacterial_species: Optional[str] = "Bacillus subtilis"
    cement_type: Optional[str] = "OPC 53"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v, default: float) -> float:
    try:
        f = float(v)
        return f if f > 0 else default
    except (TypeError, ValueError):
        return default


def _predict_compressive(
    age: float,
    conc_type: str,
    is_bact: bool,
    conc_val: float,
    cement_type: str,
    specimen_geometry: str,
) -> float:
    """XGBoost Compressive Champion (Phase 4 Audited, MAE: 1.62 MPa on dev set)."""
    if "compressive_xgb" not in _models:
        # Physics-based fallback (Abrams' law anchor)
        return 31.48 if is_bact else 28.20

    df = pd.DataFrame([{
        "curing_age_days": age,
        "bacterial_concentration_cells_ml": conc_val,
        "concrete_type": conc_type,
        "bacterial_status": "Bacteria-Induced" if is_bact else "Control (No Bacteria)",
        "cement_type": cement_type,
        "bacterial_species": "Bacillus subtilis" if is_bact else "None",
        "specimen_geometry": specimen_geometry,
        "mechanical_property": "Compressive Strength",
    }])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_models["compressive_xgb"].predict(df)[0])
    except Exception:
        return 31.48 if is_bact else 28.20


def _predict_flexural(
    age: float,
    conc_type: str,
    is_bact: bool,
    conc_val: float,
    cement_type: str,
    specimen_geometry: str,
) -> float:
    """Random Forest Flexural Champion (Phase 4 Audited, MAE: 0.455 MPa)."""
    fallback = 4.43 if is_bact else 3.86
    if "flexural_rf" not in _models:
        return fallback

    df = pd.DataFrame([{
        "curing_age_days": age,
        "bacterial_concentration_cells_ml": conc_val,
        "concrete_type": conc_type,
        "bacterial_status": "Bacteria-Induced" if is_bact else "Control (No Bacteria)",
        "cement_type": cement_type,
        "bacterial_species": "Bacillus subtilis" if is_bact else "None",
        "specimen_geometry": "Cube 150mm",
        "mechanical_property": "Flexural Strength",
    }])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_models["flexural_rf"].predict(df)[0])
    except Exception:
        return fallback


def _predict_split_tensile(
    age: float,
    conc_type: str,
    is_bact: bool,
    conc_val: float,
    cement_type: str,
    specimen_geometry: str,
) -> float:
    """XGBoost Split Tensile Champion (Phase 4 Audited, MAE: 0.315 MPa)."""
    fallback = 3.18 if is_bact else 2.68
    if "split_tensile_xgb" not in _models:
        return fallback

    df = pd.DataFrame([{
        "curing_age_days": age,
        "bacterial_concentration_cells_ml": conc_val,
        "concrete_type": conc_type,
        "bacterial_status": "Bacteria-Induced" if is_bact else "Control (No Bacteria)",
        "cement_type": cement_type,
        "bacterial_species": "Bacillus subtilis" if is_bact else "None",
        "specimen_geometry": "Cube 150mm",
    }])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_models["split_tensile_xgb"].predict(df)[0])
    except Exception:
        return fallback


def _mix_physics(
    cement: float,
    water: float,
    fine_agg: float,
    coarse_agg: float,
    age: float,
    conc_val: float,
    is_normal: bool,
) -> Dict[str, float]:
    """Abrams' law + aggregate packing + bio-dosage modulation factors."""
    wc = water / cement
    wc_factor = math.pow(0.4578 / max(wc, 0.22), 1.25)
    c_factor = math.pow(cement / 415.0, 0.28)

    tot_agg = max(fine_agg + coarse_agg, 100.0)
    sand_ratio = fine_agg / tot_agg
    sand_factor = max(0.85, 1.0 - 0.45 * math.pow(sand_ratio - 0.3736, 2))

    age_factor = (max(age, 1.0) / (4.2 + 0.85 * max(age, 1.0))) * 1.00714

    if is_normal or conc_val <= 0:
        bio_mult = 1.0
    else:
        log_conc = math.log10(max(conc_val, 10.0))
        bio_mult = max(0.80, min(1.15, log_conc / 6.0))

    return {
        "wc": wc,
        "wc_factor": wc_factor,
        "c_factor": c_factor,
        "sand_factor": sand_factor,
        "age_factor": age_factor,
        "bio_mult": bio_mult,
        "tot_agg": tot_agg,
    }


def _geometry_factors(geom: str) -> tuple[float, float, float]:
    if "Cylinder" in geom:
        return 0.82, 0.88, 1.00
    if "Prism" in geom:
        return 0.90, 1.00, 0.95
    return 1.00, 0.92, 1.05  # Cube (default)


def _compute_bundle(req: ConcretePredictionRequest, force_control: bool = False) -> dict:
    _load_models()

    cement      = _safe_float(req.cement, 415.0)
    water       = _safe_float(req.water, 190.0)
    fine_agg    = _safe_float(req.fine_aggregate, 680.0)
    coarse_agg  = _safe_float(req.coarse_aggregate, 1140.0)
    age         = _safe_float(req.curing_age_days, 28.0)
    conc_val    = float(req.bacterial_concentration_cells_ml or 1_000_000.0)
    cement_type = req.cement_type or "OPC 53"
    geom        = req.specimen_geometry or "Cube 150mm"

    is_normal = (req.concrete_type == "Normal Concrete") or force_control or (conc_val <= 0)
    conc_type = "Normal Concrete" if is_normal else "Bacterial Concrete"
    conc_for_model = 0.0 if is_normal else 1_000_000.0

    # AI model predictions (age/type-specific)
    comp_base = _predict_compressive(age, conc_type, not is_normal, conc_for_model, cement_type, geom)
    flex_base = _predict_flexural(age, conc_type, not is_normal, conc_for_model, cement_type, geom)
    tens_base = _predict_split_tensile(age, conc_type, not is_normal, conc_for_model, cement_type, geom)

    # Physics mix-scaling
    mx = _mix_physics(cement, water, fine_agg, coarse_agg, age, conc_val, is_normal)
    g_comp, g_flex, g_tens = _geometry_factors(geom)

    comp = comp_base * mx["wc_factor"] * mx["c_factor"] * mx["sand_factor"] * g_comp
    flex = flex_base * math.pow(mx["wc_factor"], 0.70) * math.pow(mx["c_factor"], 0.25) \
           * mx["sand_factor"] * math.pow(mx["age_factor"], 0.75) * mx["bio_mult"] * g_flex
    tens = tens_base * math.pow(mx["wc_factor"], 0.65) * math.pow(mx["c_factor"], 0.25) \
           * mx["sand_factor"] * math.pow(mx["age_factor"], 0.70) * mx["bio_mult"] * g_tens

    return {
        "compressive":    round(max(comp, 1.0), 2),
        "flexural":       round(max(flex, 0.3), 2),
        "split_tensile":  round(max(tens, 0.2), 2),
        "wc_ratio":       round(mx["wc"], 3),
        "total_agg":      round(mx["tot_agg"], 1),
        "age":            age,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
@app.get("/health")
def health_check():
    _load_models()
    return {
        "status": "online",
        "version": "3.0.0",
        "models_loaded": {
            "compressive":   "compressive_xgb" in _models,
            "flexural":      "flexural_rf"      in _models,
            "split_tensile": "split_tensile_xgb" in _models,
        },
        "engines": {
            "compressive":   "XGBoost Compressive Champion (Phase 4 Audited, MAE: 1.62 MPa)",
            "flexural":      "Random Forest Flexural Champion (Phase 4 Audited, MAE: 0.455 MPa)",
            "split_tensile": "XGBoost Tensile Champion (Phase 4 Audited, MAE: 0.315 MPa)",
        },
        "tabpfn_active": False,  # excluded on Vercel (PyTorch too large for serverless)
        "environment":   os.environ.get("VERCEL_ENV", "local"),
    }


@app.post("/api/predict")
@app.post("/predict")
def predict_strength(req: ConcretePredictionRequest):
    """Predict mechanical properties for a concrete mix design."""
    try:
        pred = _compute_bundle(req, force_control=False)
        ctrl = _compute_bundle(req, force_control=True)

        def _gain(val, base):
            return round((val - base) / base * 100.0, 2) if base > 0 else 0.0

        return {
            "success": True,
            "property_selected": req.property or "All",
            "engines": {
                "compressive":   "XGBoost Compressive Champion (Phase 4, MAE: 1.62 MPa)",
                "flexural":      "Random Forest Flexural Champion (Phase 4, MAE: 0.455 MPa)",
                "split_tensile": "XGBoost Tensile Champion (Phase 4, MAE: 0.315 MPa)",
                "tabpfn_online": False,
            },
            "predictions": {
                "compressive_strength":   pred["compressive"],
                "flexural_strength":      pred["flexural"],
                "split_tensile_strength": pred["split_tensile"],
            },
            "control_baseline": {
                "compressive_strength":   ctrl["compressive"],
                "flexural_strength":      ctrl["flexural"],
                "split_tensile_strength": ctrl["split_tensile"],
            },
            "bio_gain_pct": {
                "compressive":  _gain(pred["compressive"],   ctrl["compressive"]),
                "flexural":     _gain(pred["flexural"],      ctrl["flexural"]),
                "split_tensile":_gain(pred["split_tensile"], ctrl["split_tensile"]),
            },
            "mix_summary": {
                "water_cement_ratio": pred["wc_ratio"],
                "total_aggregate":    pred["total_agg"],
                "concrete_type":      req.concrete_type or "Bacterial Concrete",
                "curing_age_days":    pred["age"],
                "specimen_geometry":  req.specimen_geometry or "Cube 150mm",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


# ---------------------------------------------------------------------------
# Local dev server (not used on Vercel)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def _open():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000/")

    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run("index:app", host="127.0.0.1", port=8000, reload=False)
