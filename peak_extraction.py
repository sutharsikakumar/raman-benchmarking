import json
import sys
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths, medfilt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    n = len(y)
    D_sparse = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csr")
    H = lam * D_sparse.T.dot(D_sparse)
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = diags(w, format="csr")
        z = spsolve(W + H, w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def preprocess(y):
    y_work = y.copy().astype(float)

    y_med = medfilt(y_work, kernel_size=5)
    diff = np.abs(y_work - y_med)
    thresh = 3.0 * np.std(diff)
    y_work[diff > thresh] = y_med[diff > thresh]

    baseline = als_baseline(y_work, lam=1e5, p=0.01, niter=10)
    y_work = np.clip(y_work - baseline, 0, None)

    win = 15
    if win % 2 == 0:
        win -= 1
    win = min(win, len(y_work) - 1 if len(y_work) % 2 == 0 else len(y_work))
    y_work = np.clip(savgol_filter(y_work, window_length=win, polyorder=3), 0, None)

    p95 = float(np.percentile(y_work, 95))
    if p95 > 0:
        y_work = y_work / p95

    return y_work


def compute_qc(y):
    mean_val = float(np.mean(y))
    peak_val = float(np.max(y))
    noise_region = y[:20] if len(y) > 20 else y
    noise_std = float(np.std(noise_region))
    peaks_idx, _ = find_peaks(y, prominence=0.02)
    return {
        "snr_proxy":       round(float(peak_val / noise_std) if noise_std > 0 else 0.0, 2),
        "peak_mean_ratio": round(float(peak_val / mean_val) if mean_val > 0 else 0.0, 2),
        "num_peaks":       int(len(peaks_idx)),
    }


REGIONS = {
    "100_300":   (100,  300),
    "300_500":   (300,  500),
    "500_900":   (500,  900),
    "1300_1400": (1300, 1400),
    "1550_1620": (1550, 1620),
    "2650_2750": (2650, 2750),
}


def region_fractions(x, y):
    total = float(np.trapezoid(y, x)) if len(x) > 1 else float(np.sum(y))
    if total <= 0:
        return {k: 0.0 for k in REGIONS}
    fracs = {}
    for label, (lo, hi) in REGIONS.items():
        mask = (x >= lo) & (x <= hi)
        area = float(np.trapezoid(y[mask], x[mask])) if mask.sum() > 1 else float(np.sum(y[mask]))
        fracs[label] = round(max(area, 0.0) / total, 5)
    return fracs


def extract_peaks(x, y, prominence=0.03, min_height=0.02):
    peak_idx, props = find_peaks(y, prominence=prominence, height=min_height)
    if len(peak_idx) == 0:
        return []
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    widths_samples, _, left_ips, right_ips = peak_widths(y, peak_idx, rel_height=0.5)
    peaks_out = []
    for i, idx in enumerate(peak_idx):
        fwhm_cm1  = float(widths_samples[i]) * dx
        left_cm1  = float(np.interp(left_ips[i],  np.arange(len(x)), x))
        right_cm1 = float(np.interp(right_ips[i], np.arange(len(x)), x))
        peaks_out.append({
            "cm1":        round(float(x[idx]), 1),
            "rel_height": round(float(y[idx]), 5),
            "prominence": round(float(props["prominences"][i]), 5),
            "fwhm_cm1":   round(fwhm_cm1, 2),
            "left_half":  round(left_cm1, 1),
            "right_half": round(right_cm1, 1),
        })
    peaks_out.sort(key=lambda p: p["rel_height"], reverse=True)
    return peaks_out


def run_pipeline(spectrum_json):
    x = np.array(spectrum_json["x"], dtype=float)
    y = np.array(spectrum_json["y"], dtype=float)
    spectrum_id = spectrum_json.get("spectrum_id", "unknown")

    y_denoised = preprocess(y)

    return {
        "spectrum_id":      spectrum_id,
        "x_unit":           spectrum_json.get("x_unit", "cm^-1"),
        "y_unit":           spectrum_json.get("y_unit", "counts"),
        "x_range_cm1":      [round(float(x[0]), 1), round(float(x[-1]), 1)],
        "qc":               compute_qc(y_denoised),
        "region_fractions": region_fractions(x, y_denoised),
        "peaks":            extract_peaks(x, y_denoised),
        "signals": {
            "x":          [round(float(v), 2) for v in x],
            "y_original": [round(float(v), 6) for v in y],
            "y_denoised": [round(float(v), 6) for v in y_denoised],
        },
    }


def plot(result, save_path=None):
    import matplotlib.pyplot as plt

    x          = np.array(result["signals"]["x"])
    y_orig     = np.array(result["signals"]["y_original"])
    y_denoised = np.array(result["signals"]["y_denoised"])
    peaks      = result["peaks"]

    y_orig_norm = y_orig / (float(np.percentile(y_orig, 95)) or 1.0)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, y_orig_norm, color="#b0b8c8", linewidth=0.9, alpha=0.7, label="Original (p95-norm)")
    ax.plot(x, y_denoised,  color="#2563eb", linewidth=1.4, label="Denoised")

    for p in peaks:
        ax.axvline(p["cm1"], color="#ef4444", linewidth=0.6, alpha=0.5, linestyle="--")
        ax.annotate(
            f"{p['cm1']:.0f}",
            xy=(p["cm1"], p["rel_height"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#dc2626",
        )

    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (normalised)", fontsize=11)
    ax.set_title(f"Raman Spectrum  —  {result['spectrum_id']}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, linewidth=0.4, alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python raman_preprocess.py <spectrum.json> [output.json] [--plot plot.png]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        spectrum_json = json.load(f)

    result = run_pipeline(spectrum_json)

    plot_path   = None
    output_path = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--plot":
            plot_path = args[i + 1] if i + 1 < len(args) else "plot.png"
            i += 2
        else:
            output_path = args[i]
            i += 1

    if output_path is None:
        output_path = f"{result.get('spectrum_id', 'output')}_preprocessed.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written → {output_path}")

    plot(result, save_path=plot_path)


if __name__ == "__main__":
    main()