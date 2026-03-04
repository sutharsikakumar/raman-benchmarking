import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.signal import savgol_filter, medfilt, find_peaks, peak_widths
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import pywt
    PYWT_OK = True
except Exception:
    PYWT_OK = False


def _as_array(v):
    return np.asarray(v, dtype=float)


def _rolling_quantile_baseline(y: np.ndarray, window: int = 101, q: float = 0.10) -> np.ndarray:
    n = y.size
    w = int(window)
    if w < 11:
        w = 11
    if w % 2 == 0:
        w += 1
    h = w // 2
    yp = np.pad(y, (h, h), mode="edge")
    out = np.empty(n, dtype=float)
    qp = q * 100.0
    for i in range(n):
        out[i] = np.percentile(yp[i:i + w], qp)
    return out


def _als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    if not SCIPY_OK:
        return _rolling_quantile_baseline(y, window=max(101, (len(y) // 40) * 2 + 1), q=0.10)
    L = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T.dot(D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _wavelet_denoise(y: np.ndarray, wavelet: str = "db4", level: Optional[int] = None, mode: str = "soft") -> np.ndarray:
    if not PYWT_OK:
        return y
    coeffs = pywt.wavedec(y, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else 0.0
    uthresh = sigma * math.sqrt(2.0 * math.log(max(len(y), 2)))
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, value=uthresh, mode=mode))
    rec = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
    return rec[: len(y)]


def _moving_average(y: np.ndarray, k: int = 7) -> np.ndarray:
    k = int(k)
    if k < 3:
        return y
    if k % 2 == 0:
        k += 1
    if y.size < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def _median_filter(y: np.ndarray, k: int = 5) -> np.ndarray:
    k = int(k)
    if k < 3:
        return y
    if k % 2 == 0:
        k += 1
    if y.size < k:
        return y
    if SCIPY_OK:
        return medfilt(y, kernel_size=k)
    yp = np.pad(y, (k // 2, k // 2), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(yp[i:i + k])
    return out


def _normalize_robust(y: np.ndarray) -> np.ndarray:
    s = np.percentile(y, 95)
    if not np.isfinite(s) or s == 0:
        s = np.max(np.abs(y)) + 1e-9
    return y / (s + 1e-12)


def _get_xy(spectrum: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    x = spectrum.get("x") or spectrum.get("wavenumber") or spectrum.get("wavenumbers")
    y = spectrum.get("y") or spectrum.get("intensity") or spectrum.get("counts")
    if x is None or y is None:
        raise ValueError("Spectrum JSON must contain x/y arrays (keys: x,y or wavenumber,intensity or wavenumbers,counts).")
    return list(x), list(y)


def _infer_material_from_classification(cls: Dict[str, Any]) -> str:
    for key in ("material", "label"):
        v = cls.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    pred = cls.get("prediction")
    if isinstance(pred, dict):
        v = pred.get("material") or pred.get("label")
        if isinstance(v, str) and v.strip():
            return v.strip()
        top1 = pred.get("top1")
        if isinstance(top1, dict):
            v = top1.get("label") or top1.get("material")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _best_denoise_plan(material: str) -> Dict[str, Any]:
    m = material.lower().strip()
    if m == "mos2":
        return {"method": "ALS+median5+savgol(21,3)", "baseline": ("ALS", {"lam": 1e6, "p": 0.01, "niter": 10}), "median_k": 5, "savgol": (21, 3)}
    if m in ("wse2", "wse₂"):
        return {"method": "ALS+median5+savgol(19,3)", "baseline": ("ALS", {"lam": 8e5, "p": 0.01, "niter": 10}), "median_k": 5, "savgol": (19, 3)}
    if m in ("hbn", "hexagonal boron nitride", "boron nitride"):
        return {"method": "ALS+median3+savgol(11,3)", "baseline": ("ALS", {"lam": 1e6, "p": 0.01, "niter": 10}), "median_k": 3, "savgol": (11, 3)}
    if m == "graphene":
        return {"method": "ALS+median5+savgol(17,3)", "baseline": ("ALS", {"lam": 2e6, "p": 0.01, "niter": 10}), "median_k": 5, "savgol": (17, 3)}
    if m in ("ti3c2ti", "ti3c2tx", "tmc", "mxene", "ti₃c₂tₓ", "ti3c2t_x"):
        return {"method": "quantile+wavelet(db4,soft)+movavg(9)", "baseline": ("quantile", {"window": 151, "q": 0.10}), "wavelet": ("db4", None, "soft"), "movavg_k": 9}
    return {"method": "ALS+median5+savgol(21,3)", "baseline": ("ALS", {"lam": 1e6, "p": 0.01, "niter": 10}), "median_k": 5, "savgol": (21, 3)}


def _apply_plan(x: np.ndarray, y: np.ndarray, plan: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y0 = y.astype(float)

    baseline = np.zeros_like(y0)
    btype, bcfg = plan.get("baseline", (None, None))
    if btype == "ALS":
        baseline = _als_baseline(y0, lam=float(bcfg["lam"]), p=float(bcfg["p"]), niter=int(bcfg["niter"]))
    elif btype == "quantile":
        baseline = _rolling_quantile_baseline(y0, window=int(bcfg["window"]), q=float(bcfg["q"]))

    y1 = y0 - baseline

    mk = plan.get("median_k")
    if mk is not None:
        y1 = _median_filter(y1, int(mk))

    wcfg = plan.get("wavelet")
    if wcfg is not None:
        wname, level, mode = wcfg
        y1 = _wavelet_denoise(y1, wavelet=str(wname), level=level, mode=str(mode))

    sg = plan.get("savgol")
    if sg is not None:
        win, poly = int(sg[0]), int(sg[1])
        if win % 2 == 0:
            win += 1
        if SCIPY_OK and y1.size >= max(win, poly + 2):
            y1 = savgol_filter(y1, window_length=win, polyorder=poly)
        else:
            y1 = _moving_average(y1, k=max(7, min(21, win)))

    mak = plan.get("movavg_k")
    if mak is not None:
        y1 = _moving_average(y1, int(mak))

    meta = {
        "method": plan.get("method", ""),
        "scipy": bool(SCIPY_OK),
        "pywt": bool(PYWT_OK),
        "baseline": {"type": btype, **(bcfg or {})},
    }
    return x, y1, meta


def _extract_peaks(x: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    yn = _normalize_robust(y)
    if SCIPY_OK:
        prom = max(0.02, 0.15 * float(np.std(yn)))
        pidx, props = find_peaks(yn, prominence=prom, distance=max(5, len(yn) // 400))
        if pidx.size == 0:
            return []
        order = np.argsort(props.get("prominences", yn[pidx]))[::-1]
        pidx = pidx[order][:25]
        widths = peak_widths(yn, pidx, rel_height=0.5)
        out = []
        dx = float(np.median(np.diff(x))) if x.size > 1 else float("nan")
        for j, i in enumerate(pidx):
            out.append({
                "cm1": float(x[i]),
                "rel_height": float(yn[i]),
                "prominence": float(props["prominences"][order][j]) if "prominences" in props else float("nan"),
                "fwhm_cm1": float(widths[0][j] * dx) if np.isfinite(dx) else float("nan"),
            })
        out.sort(key=lambda d: d["cm1"])
        return out
    thr = 0.20
    idx = []
    for i in range(1, len(yn) - 1):
        if yn[i] >= thr and yn[i] >= yn[i - 1] and yn[i] >= yn[i + 1]:
            idx.append(i)
    if not idx:
        return []
    idx = np.array(idx, dtype=int)
    idx = idx[np.argsort(yn[idx])[::-1]][:25]
    out = [{"cm1": float(x[i]), "rel_height": float(yn[i]), "prominence": float("nan"), "fwhm_cm1": float("nan")} for i in idx]
    out.sort(key=lambda d: d["cm1"])
    return out


def _region_fractions(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    regions = {
        "100_300": (100.0, 300.0),
        "300_500": (300.0, 500.0),
        "800_1100": (800.0, 1100.0),
        "1300_1400": (1300.0, 1400.0),
        "1550_1620": (1550.0, 1620.0),
        "2650_2750": (2650.0, 2750.0),
    }
    total = float(np.trapezoid(np.abs(y), x)) if x.size > 1 else float(np.sum(np.abs(y)))
    out = {}
    for k, (lo, hi) in regions.items():
        m = (x >= lo) & (x <= hi)
        if not np.any(m) or total <= 0:
            out[k] = 0.0
        else:
            out[k] = float(np.trapezoid(np.abs(y[m]), x[m]) / total)
    return out


def _qc(y: np.ndarray) -> Dict[str, float]:
    y = y.astype(float)
    if y.size < 10:
        return {"snr_proxy": 0.0, "peak_mean_ratio": float("nan")}
    p95 = float(np.percentile(y, 95))
    p05 = float(np.percentile(y, 5))
    noise = float(np.median(np.abs(y - np.median(y))) + 1e-12)
    snr = float((p95 - p05) / noise) if noise > 0 else 0.0
    pmr = float((np.max(y) + 1e-12) / (np.mean(np.abs(y)) + 1e-12))
    return {"snr_proxy": snr, "peak_mean_ratio": pmr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spectrum_json")
    ap.add_argument("classification_json")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    with open(args.spectrum_json, "r", encoding="utf-8") as f:
        spectrum = json.load(f)

    with open(args.classification_json, "r", encoding="utf-8") as f:
        cls = json.load(f)

    material = _infer_material_from_classification(cls)
    if not material:
        raise ValueError("Classification JSON must contain a material label (key: material or prediction.top1.label).")

    x_list, y_list = _get_xy(spectrum)
    x = _as_array(x_list)
    y = _as_array(y_list)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    plan = _best_denoise_plan(material)
    xd, yd, denoise_meta = _apply_plan(x, y, plan)

    out = {}
    out["spectrum_id"] = spectrum.get("spectrum_id") or spectrum.get("metadata", {}).get("id") or os.path.basename(args.spectrum_json)
    out["source_spectrum_file"] = spectrum.get("source_file") or spectrum.get("file") or os.path.basename(args.spectrum_json)
    out["source_classification_file"] = cls.get("source_file") or cls.get("file") or os.path.basename(args.classification_json)
    out["x_unit"] = spectrum.get("x_unit", "cm^-1")
    out["y_unit"] = spectrum.get("y_unit", "counts")
    out["x_range_cm1"] = [float(np.min(xd)), float(np.max(xd))]
    out["denoise"] = denoise_meta
    out["qc"] = _qc(yd)
    out["region_fractions"] = _region_fractions(xd, yd)
    out["peaks"] = _extract_peaks(xd, yd)
    out["signals"] = {
        "y_denoised": [float(v) for v in yd.tolist()]
    }
    if "confidence" in cls:
        try:
            out["confidence"] = float(cls["confidence"])
        except Exception:
            pass
    if "key_evidence" in cls:
        out["key_evidence"] = cls["key_evidence"]
    if "reasoning" in cls:
        out["reasoning"] = cls["reasoning"]
    if "eliminated_candidates" in cls:
        out["eliminated_candidates"] = cls["eliminated_candidates"]

    if args.output is None:
        base, _ = os.path.splitext(args.spectrum_json)
        out_path = base + ".benchmark.json"
    else:
        out_path = args.output

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()