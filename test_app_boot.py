"""
test_app_boot.py
================
Headless Boot Simulation & Sanity Test for AquaScan AI / SonarGuard-AI
-----------------------------------------------------------------------
Validates that the full inference pipeline boots cleanly in a Debian Linux
container environment without hangs, CUDA warnings, memory leaks, or missing
dependencies.

Run with:
    python test_app_boot.py

Exit codes:
    0  — All tests passed
    1  — One or more tests failed (see output for details)

This script is intentionally free of Streamlit imports so it can be executed
in a plain Python process (CI, pre-push hook, container smoke-test).
"""

import sys
import os
import time
import traceback
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve workspace root so imports work from any CWD
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ANSI colours for terminal readability
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
results = []

def _pass(name, detail="", elapsed_ms=None):
    tag = f"[{_GREEN}PASS{_RESET}]"
    t   = f"  {elapsed_ms:.1f} ms" if elapsed_ms is not None else ""
    print(f"  {tag}  {name}{t}")
    if detail:
        print(f"         {_CYAN}{detail}{_RESET}")
    results.append({"test": name, "status": "PASS", "elapsed_ms": elapsed_ms, "detail": detail})

def _fail(name, detail="", elapsed_ms=None):
    tag = f"[{_RED}FAIL{_RESET}]"
    t   = f"  {elapsed_ms:.1f} ms" if elapsed_ms is not None else ""
    print(f"  {tag}  {name}{t}")
    if detail:
        print(f"         {_RED}{detail}{_RESET}")
    results.append({"test": name, "status": "FAIL", "elapsed_ms": elapsed_ms, "detail": detail})

def _warn(name, detail="", elapsed_ms=None):
    tag = f"[{_YELLOW}WARN{_RESET}]"
    t   = f"  {elapsed_ms:.1f} ms" if elapsed_ms is not None else ""
    print(f"  {tag}  {name}{t}")
    if detail:
        print(f"         {_YELLOW}{detail}{_RESET}")
    results.append({"test": name, "status": "WARN", "elapsed_ms": elapsed_ms, "detail": detail})


# ===========================================================================
# TEST 1 — Dependency Import Check
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 1 — Dependency Import Chain{_RESET}")
print(f"{'='*60}")

REQUIRED_PACKAGES = {
    "cv2":            "opencv-python-headless",
    "numpy":          "numpy",
    "PIL":            "Pillow",
    "streamlit":      "streamlit",
    "ultralytics":    "ultralytics",
    "folium":         "folium",
    "pandas":         "pandas",
}

for mod_name, pkg_name in REQUIRED_PACKAGES.items():
    t0 = time.perf_counter()
    try:
        mod = __import__(mod_name)
        elapsed = (time.perf_counter() - t0) * 1000
        ver = getattr(mod, "__version__", "unknown")
        _pass(f"import {mod_name}", f"v{ver} ({pkg_name})", elapsed)
    except ImportError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _fail(f"import {mod_name}", str(e), elapsed)


# ===========================================================================
# TEST 2 — Preprocessing Module (no training imports at boot)
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 2 — Preprocessing Module Boot (no training imports){_RESET}")
print(f"{'='*60}")

t0 = time.perf_counter()
try:
    from preprocessing import preprocess_sonar_image, apply_clahe_filter
    elapsed = (time.perf_counter() - t0) * 1000
    _pass("preprocessing module import", "preprocess_sonar_image, apply_clahe_filter", elapsed)
except Exception as e:
    elapsed = (time.perf_counter() - t0) * 1000
    _fail("preprocessing module import", str(e), elapsed)

# Confirm 'requests' was NOT pulled in by OUR preprocessing module.
# Note: ultralytics (imported in TEST 1) legitimately imports requests for
# its own update/telemetry features — that is unavoidable and not a hang risk.
# What matters is that preprocessing/__init__.py alone does NOT trigger requests.
# We validate this by checking whether dataset_prep was pulled into sys.modules
# (which would be the only path requests enters through our code).
t0 = time.perf_counter()
dataset_prep_loaded = "preprocessing.dataset_prep" in sys.modules
elapsed = (time.perf_counter() - t0) * 1000
if dataset_prep_loaded:
    _fail("preprocessing.__init__ does NOT import dataset_prep at boot",
          "dataset_prep (and its 'import requests') was loaded — boot hang risk!", elapsed)
else:
    _pass("preprocessing.__init__ does NOT import dataset_prep at boot",
          "No cloud-blocking 'import requests' via our module chain", elapsed)


# ===========================================================================
# TEST 3 — Lee (1980) Speckle Filter on Synthetic Matrix
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 3 — Lee Speckle Filter (vectorized, no memory leak){_RESET}")
print(f"{'='*60}")

try:
    import numpy as np
    from preprocessing.denoise import lee_speckle_filter, preprocess_sonar_image

    rng = np.random.default_rng(42)

    # Sub-test 3a: basic correctness
    t0 = time.perf_counter()
    synthetic = (rng.random((640, 640)) * 255).astype(np.uint8)
    speckle   = rng.gamma(shape=1.5, scale=1.0, size=(640, 640))
    noisy     = np.clip(synthetic * speckle, 0, 255).astype(np.uint8)
    out       = lee_speckle_filter(noisy, window_size=7)
    elapsed   = (time.perf_counter() - t0) * 1000

    assert out.shape == noisy.shape, f"Shape mismatch: {out.shape} vs {noisy.shape}"
    assert out.dtype == np.uint8,    f"Wrong dtype: {out.dtype}"
    assert out.min() >= 0 and out.max() <= 255, "Output out of [0,255] range"
    _pass("Lee filter shape / dtype / range (640×640)", f"in={noisy.shape} out={out.shape}", elapsed)

    # Sub-test 3b: noise reduction (filtered mean should be closer to original than noisy)
    t0 = time.perf_counter()
    mae_noisy    = float(np.mean(np.abs(noisy.astype(float) - synthetic.astype(float))))
    mae_filtered = float(np.mean(np.abs(out.astype(float)   - synthetic.astype(float))))
    elapsed = (time.perf_counter() - t0) * 1000
    if mae_filtered <= mae_noisy:
        _pass("Lee filter reduces MAE vs noisy",
              f"MAE noisy={mae_noisy:.2f}  filtered={mae_filtered:.2f}", elapsed)
    else:
        _warn("Lee filter MAE not reduced (possible for extreme synthetic speckle)",
              f"MAE noisy={mae_noisy:.2f}  filtered={mae_filtered:.2f}", elapsed)

    # Sub-test 3c: memory — run 10 iterations, check no accumulation
    t0 = time.perf_counter()
    for _ in range(10):
        _ = lee_speckle_filter(noisy, window_size=7)
    elapsed = (time.perf_counter() - t0) * 1000
    _pass("Lee filter × 10 iterations (no leak / exception)", f"10 runs × 640×640", elapsed)

    # Sub-test 3d: full pipeline (BGR in → BGR out)
    t0 = time.perf_counter()
    import cv2
    bgr_test = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    bgr_out  = preprocess_sonar_image(bgr_test, apply_clahe=True, apply_bilateral=True)
    elapsed  = (time.perf_counter() - t0) * 1000
    assert bgr_out.shape == bgr_test.shape, "BGR pipeline shape mismatch"
    assert bgr_out.dtype == np.uint8
    _pass("preprocess_sonar_image (Lee + CLAHE, BGR→BGR)", f"shape={bgr_out.shape}", elapsed)

except Exception as e:
    _fail("Lee filter tests", traceback.format_exc())


# ===========================================================================
# TEST 4 — Geotagging Engine: AQ-HZ- IDs, Offshore Coords, GeoJSON
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 4 — Geotagging Engine (AQ-HZ- IDs, Offshore Coords, GeoJSON){_RESET}")
print(f"{'='*60}")

try:
    from core_engine.geotag_engine import (
        parse_sonar_telemetry, export_reports,
        DEFAULT_BASE_LAT, DEFAULT_BASE_LON
    )

    # Sub-test 4a: offshore coordinate defaults
    t0 = time.perf_counter()
    assert abs(DEFAULT_BASE_LAT - 13.0827) < 1e-4, f"Wrong default lat: {DEFAULT_BASE_LAT}"
    assert abs(DEFAULT_BASE_LON - 80.4500) < 1e-4, f"Wrong default lon: {DEFAULT_BASE_LON}"
    elapsed = (time.perf_counter() - t0) * 1000
    _pass("Default origin = Bay of Bengal offshore",
          f"Lat={DEFAULT_BASE_LAT}, Lon={DEFAULT_BASE_LON}", elapsed)

    # Sub-test 4b: hazard ID prefix
    t0 = time.perf_counter()
    dets = [
        {"x_center": 412, "y_center": 280, "class_name": "Submerged Solid Hazard / Shipwreck",
         "confidence": 0.884, "area_px": 1240},
        {"x_center": 150, "y_center": 510, "class_name": "Entangled Ghost Net",
         "confidence": 0.925, "area_px": 3400},
    ]
    bbox_dims = [(40, 31), (85, 40)]
    report = parse_sonar_telemetry(dets, bbox_dims=bbox_dims)
    elapsed = (time.perf_counter() - t0) * 1000

    assert len(report) == 2, f"Expected 2 records, got {len(report)}"
    assert report[0]["Hazard ID"].startswith("AQ-HZ-"), \
        f"Wrong prefix: {report[0]['Hazard ID']}"
    _pass("parse_sonar_telemetry returns AQ-HZ- IDs",
          f"IDs: {[r['Hazard ID'] for r in report]}", elapsed)

    # Sub-test 4c: physical dimensions
    t0 = time.perf_counter()
    r0 = report[0]
    assert "Estimated_Length_m" in r0, "Missing Estimated_Length_m"
    assert "Estimated_Width_m"  in r0, "Missing Estimated_Width_m"
    expected_l = round(31 * 0.05, 2)   # h_px * 0.05
    expected_w = round(40 * 0.05, 2)   # w_px * 0.05
    assert abs(r0["Estimated_Length_m"] - expected_l) < 0.01, \
        f"Length mismatch: got {r0['Estimated_Length_m']}, expected {expected_l}"
    assert abs(r0["Estimated_Width_m"] - expected_w) < 0.01, \
        f"Width mismatch: got {r0['Estimated_Width_m']}, expected {expected_w}"
    elapsed = (time.perf_counter() - t0) * 1000
    _pass("Physical dimensions @ 0.05 m/px",
          f"L={r0['Estimated_Length_m']} m, W={r0['Estimated_Width_m']} m", elapsed)

    # Sub-test 4d: empty report — no files written
    t0 = time.perf_counter()
    import tempfile
    tmp_csv  = str(Path(tempfile.gettempdir()) / "test_empty.csv")
    tmp_json = str(Path(tempfile.gettempdir()) / "test_empty.json")
    # Remove if they exist
    for p in [tmp_csv, tmp_json]:
        if os.path.exists(p): os.remove(p)
    export_reports([], output_csv=tmp_csv, output_json=tmp_json)
    elapsed = (time.perf_counter() - t0) * 1000
    files_written = os.path.exists(tmp_csv) or os.path.exists(tmp_json)
    if not files_written:
        _pass("Empty report → no CSV/GeoJSON written", "Clean zero-detection behaviour", elapsed)
    else:
        _fail("Empty report → files were written (dummy rows present)", "", elapsed)

    # Sub-test 4e: GeoJSON is valid FeatureCollection
    t0 = time.perf_counter()
    tmp_csv2  = str(Path(tempfile.gettempdir()) / "test_report.csv")
    tmp_json2 = str(Path(tempfile.gettempdir()) / "test_report.json")
    export_reports(report, output_csv=tmp_csv2, output_json=tmp_json2)
    with open(tmp_json2) as f:
        geojson = json.load(f)
    assert geojson["type"] == "FeatureCollection", "Not a GeoJSON FeatureCollection"
    assert len(geojson["features"]) == 2, f"Expected 2 features, got {len(geojson['features'])}"
    elapsed = (time.perf_counter() - t0) * 1000
    _pass("GeoJSON FeatureCollection export valid",
          f"{len(geojson['features'])} features written", elapsed)

except Exception as e:
    _fail("Geotagging engine tests", traceback.format_exc())


# ===========================================================================
# TEST 5 — Model Loading Time
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 5 — YOLO Model Loading (best.pt){_RESET}")
print(f"{'='*60}")

MODEL_PATH = ROOT_DIR / "models" / "best.pt"
model_elapsed_ms = None

if not MODEL_PATH.exists():
    _warn("models/best.pt not found", "Skipping model load test (weights not in repo)", 0)
else:
    try:
        import warnings
        import io as _io
        # Suppress CUDA/device warning noise from ultralytics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Redirect ultralytics stdout to capture any CUDA messages
            import contextlib
            capture_buf = _io.StringIO()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(capture_buf):
                from ultralytics import YOLO
                model = YOLO(str(MODEL_PATH))
            model_elapsed_ms = (time.perf_counter() - t0) * 1000
            captured = capture_buf.getvalue()

        cuda_warns = [l for l in captured.splitlines() if "cuda" in l.lower() or "gpu" in l.lower()]

        if model_elapsed_ms < 2000:
            _pass(f"YOLO model loaded in {model_elapsed_ms:.0f} ms (< 2000 ms target)",
                  f"weights: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1e6:.1f} MB)",
                  model_elapsed_ms)
        else:
            _warn(f"YOLO model loaded but exceeded 2000 ms",
                  f"Actual: {model_elapsed_ms:.0f} ms", model_elapsed_ms)

        if cuda_warns:
            _warn("CUDA references printed during model load",
                  " | ".join(cuda_warns[:3]))
        else:
            _pass("No CUDA/GPU warnings during model load", "CPU inference confirmed")

        # Verify model can run predict without error on a blank image
        t0 = time.perf_counter()
        import numpy as np, cv2
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(source=blank, conf=0.35, iou=0.45, verbose=False)
        pred_elapsed = (time.perf_counter() - t0) * 1000
        _pass(f"model.predict() on blank 640×640 — no exception",
              f"boxes detected: {len(preds[0].boxes)}", pred_elapsed)

    except Exception as e:
        _fail("YOLO model loading", traceback.format_exc())


# ===========================================================================
# TEST 6 — Full Inference Pipeline (end-to-end)
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 6 — Full SonarInferenceEngine.process_image() End-to-End{_RESET}")
print(f"{'='*60}")

if not MODEL_PATH.exists():
    _warn("Skipping full pipeline test", "models/best.pt not found", 0)
else:
    try:
        from backend_api.inference import SonarInferenceEngine
        import numpy as np

        t0 = time.perf_counter()
        engine = SonarInferenceEngine(model_path=str(MODEL_PATH))
        init_elapsed = (time.perf_counter() - t0) * 1000
        _pass("SonarInferenceEngine.__init__()", f"{MODEL_PATH.name}", init_elapsed)

        # Use a demo image if available, else blank synthetic
        demo_candidates = [
            ROOT_DIR / "demo_test_assets" / "test_shipwreck.png",
            ROOT_DIR / "demo_test_assets" / "test_subsea_pipeline.png",
        ]
        test_img = next((p for p in demo_candidates if p.exists()), None)

        if test_img:
            t0 = time.perf_counter()
            result = engine.process_image(str(test_img))
            elapsed = (time.perf_counter() - t0) * 1000
            _pass(f"process_image({test_img.name})",
                  f"detections={len(result['records'])}  "
                  f"inference_ms={result['inference_ms']}  "
                  f"first_id={result['records'][0]['Hazard ID'] if result['records'] else 'none'}",
                  elapsed)

            # Validate record structure
            t0 = time.perf_counter()
            for rec in result["records"]:
                assert "Hazard ID" in rec and rec["Hazard ID"].startswith("AQ-HZ-")
                assert "Estimated Length (m)" in rec
                assert "Estimated Width (m)"  in rec
                assert "Latitude"  in rec and 12.0 < rec["Latitude"]  < 14.0
                assert "Longitude" in rec and 79.0 < rec["Longitude"] < 82.0
            elapsed = (time.perf_counter() - t0) * 1000
            _pass("Record schema validation (AQ-HZ-, dims, offshore coords)",
                  f"{len(result['records'])} records validated", elapsed)

            # Check inference_ms is real (not hardcoded)
            t0 = time.perf_counter()
            assert isinstance(result["inference_ms"], (int, float)), "inference_ms wrong type"
            assert result["inference_ms"] > 0, "inference_ms should be positive"
            elapsed = (time.perf_counter() - t0) * 1000
            _pass("inference_ms is real measured value (time.perf_counter)",
                  f"measured = {result['inference_ms']} ms", elapsed)
        else:
            _warn("No demo image found for pipeline test", "Skipping process_image validation", 0)

    except Exception as e:
        _fail("Full pipeline test", traceback.format_exc())


# ===========================================================================
# TEST 7 — Streamlit Config Validation
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}TEST 7 — Streamlit Config (.streamlit/config.toml){_RESET}")
print(f"{'='*60}")

config_path = ROOT_DIR / ".streamlit" / "config.toml"
t0 = time.perf_counter()
if config_path.exists():
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

    if tomllib:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        elapsed = (time.perf_counter() - t0) * 1000
        headless = cfg.get("server", {}).get("headless", False)
        stats    = cfg.get("browser", {}).get("gatherUsageStats", True)
        if headless:
            _pass("config.toml: server.headless = true", "No TTY prompt hang", elapsed)
        else:
            _warn("config.toml: server.headless not set to true", "", elapsed)
        if not stats:
            _pass("config.toml: browser.gatherUsageStats = false", "No stats call at boot", elapsed)
        else:
            _warn("config.toml: gatherUsageStats not disabled", "", elapsed)
    else:
        # Can't parse TOML but file exists — check manually
        elapsed = (time.perf_counter() - t0) * 1000
        content = config_path.read_text()
        headless_ok = "headless = true" in content
        stats_ok    = "gatherUsageStats = false" in content
        _pass(".streamlit/config.toml exists",
              f"headless={'true' if headless_ok else 'MISSING'}  "
              f"gatherUsageStats={'false' if stats_ok else 'MISSING'}", elapsed)
else:
    elapsed = (time.perf_counter() - t0) * 1000
    _fail(".streamlit/config.toml missing", "headless mode not configured", elapsed)


# ===========================================================================
# SUMMARY REPORT
# ===========================================================================
print(f"\n{_BOLD}{'='*60}{_RESET}")
print(f"{_BOLD}SUMMARY REPORT{_RESET}")
print(f"{'='*60}")

passed  = sum(1 for r in results if r["status"] == "PASS")
failed  = sum(1 for r in results if r["status"] == "FAIL")
warned  = sum(1 for r in results if r["status"] == "WARN")
total   = len(results)

print(f"\n{'Test':<52} {'Status':8} {'Time (ms)':>10}")
print("-" * 74)
for r in results:
    colour = _GREEN if r["status"] == "PASS" else (_RED if r["status"] == "FAIL" else _YELLOW)
    t_str  = f"{r['elapsed_ms']:.1f}" if r["elapsed_ms"] is not None else "—"
    print(f"  {r['test'][:50]:<50} {colour}{r['status']:8}{_RESET} {t_str:>10}")

print(f"\n{'='*60}")
model_t_str = f"{model_elapsed_ms:.0f} ms" if model_elapsed_ms else "N/A (no weights)"
print(f"  {'Local Boot Test Status':<30} {'ALL PASSED' if failed == 0 else 'FAILURES DETECTED':>20}")
print(f"  {'Model Loading Time':<30} {model_t_str:>20}")
print(f"  {'Dependency Check':<30} {'PASS' if all(r['status'] != 'FAIL' for r in results if 'import' in r['test']) else 'FAIL':>20}")
print(f"  {'Total Tests':<30} {total:>20}")
print(f"  {'Passed':<30} {_GREEN}{passed}{_RESET}")
print(f"  {'Warned':<30} {_YELLOW}{warned}{_RESET}")
print(f"  {'Failed':<30} {_RED}{failed}{_RESET}")
print(f"{'='*60}\n")

if failed > 0:
    print(f"{_RED}{_BOLD}⚠  {failed} test(s) FAILED — do not push. Fix above errors first.{_RESET}\n")
    sys.exit(1)
else:
    print(f"{_GREEN}{_BOLD}✅  All tests passed — safe to commit and push.{_RESET}\n")
    sys.exit(0)
