import sys
import json
import argparse
import textwrap
import os

import anthropic
import numpy as np
from dotenv import load_dotenv

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Create a .env file with:\n"
            "ANTHROPIC_API_KEY=your_key_here\n"
            "or export it in your shell."
        )
    return api_key


DIAGNOSTIC_WINDOWS = {

    "Graphene_D":      (1310, 1390, 1350, "Graphene D band (defect-activated)"),
    "Graphene_G":      (1545, 1620, 1580, "Graphene G band (E₂g, always present)"),
    "Graphene_Dprime": (1600, 1650, 1620, "Graphene D′ band (defects, weak)"),
    "Graphene_2D":     (2620, 2780, 2700, "Graphene 2D band (key diagnostic)"),

    "hBN_E2g":         (1355, 1380, 1366, "hBN E₂g mode (single dominant peak)"),


    "MoS2_2LA_low":    (360,  380,  370,  "MoS₂ 2LA(M) low-freq shoulder"),
    "MoS2_E2g":        (378,  392,  383,  "MoS₂ E²₁g / E′ in-plane mode"),
    "MoS2_A1g":        (400,  420,  408,  "MoS₂ A₁g / A₁′ out-of-plane mode"),
    "MoS2_2LA_high":   (440,  465,  450,  "MoS₂ 2LA(M) second-order mode"),


    "WSe2_LA":         (140,  180,  160,  "WSe₂ LA(M) acoustic mode"),
    "WSe2_A1g_E2g":    (238,  272,  252,  "WSe₂ A₁g / E²₁g (near-degenerate, dominant)"),
    "WSe2_E1g":        (290,  320,  308,  "WSe₂ E₁g mode"),


    "MXene_CTiC":      (150,  250,  200,  "Ti₃C₂Tₓ C–Ti–C out-of-plane mode"),
    "MXene_TiC":       (340,  460,  400,  "Ti₃C₂Tₓ Ti–C stretching mode"),
    "MXene_TiO":       (560,  680,  620,  "Ti₃C₂Tₓ Ti–O surface group mode"),

    "MXene_broad":     (100,  900,  500,  "Ti₃C₂Tₓ broad envelope (mean intensity / flatness)"),
}

MATERIAL_WINDOWS = {
    "Graphene": ["Graphene_D", "Graphene_G", "Graphene_Dprime", "Graphene_2D"],
    "hBN":      ["hBN_E2g"],
    "MoS2":     ["MoS2_2LA_low", "MoS2_E2g", "MoS2_A1g", "MoS2_2LA_high"],
    "WSe2":     ["WSe2_LA", "WSe2_A1g_E2g", "WSe2_E1g"],
    "Ti3C2Tx":  ["MXene_CTiC", "MXene_TiC", "MXene_TiO", "MXene_broad"],
}



def summarise_spectrum(x: list, y: list) -> str:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    x_min, x_max   = float(x.min()), float(x.max())
    y_min, y_max   = float(y.min()), float(y.max())
    global_max      = float(y.max()) if y.max() != 0 else 1.0
    baseline        = float(np.percentile(y, 5))

    _CRYSTALLINE_KEYS = {k for k in DIAGNOSTIC_WINDOWS if not k.startswith("MXene")}
    _diag_peaks = []
    for _k, (_lo, _hi, _cen, _) in DIAGNOSTIC_WINDOWS.items():
        if _k not in _CRYSTALLINE_KEYS:
            continue
        _m = (x >= _lo) & (x <= _hi)
        if _m.any() and (x_min <= _cen <= x_max):
            _diag_peaks.append(float(y[_m].max()))
    diagnostic_max = max(_diag_peaks) if _diag_peaks else global_max

    pl_warning = ""
    if global_max > 3 * diagnostic_max:
        pl_warning = (
            f"\n  WARNING: global max ({global_max:.0f} cts) is "
            f"{global_max/diagnostic_max:.1f}x the strongest diagnostic window peak "
            f"({diagnostic_max:.0f} cts). A photoluminescence or substrate artifact "
            f"likely dominates the raw spectrum. Relative %% below uses diagnostic_max."
        )

    window_results: dict[str, dict] = {}

    for key, (lo, hi, centre, desc) in DIAGNOSTIC_WINDOWS.items():
        mask    = (x >= lo) & (x <= hi)
        covered = (x_min <= centre <= x_max)

        if not mask.any() or not covered:
            window_results[key] = dict(
                lo=lo, hi=hi, centre=centre, desc=desc,
                covered=covered, peak_val=None,
                peak_pos=None, mean_val=None, rel_pct=None, note=None,
            )
            continue

        ys       = y[mask]
        xs       = x[mask]
        idx_max  = int(np.argmax(ys))
        peak_val = float(ys[idx_max])
        peak_pos = float(xs[idx_max])
        mean_val = float(ys.mean())


        if key == "MXene_broad":
            peak_to_mean = peak_val / mean_val if mean_val > 0 else 0
            mean_rel_pct = mean_val / diagnostic_max * 100
            flat = peak_to_mean < 3
            note = (
                f"mean={mean_val:.1f} cts ({mean_rel_pct:.1f}% of global max), "
                f"peak/mean ratio={peak_to_mean:.1f} "
                f"({'broad/flat → TMC-consistent' if flat else 'sharp peak present → not TMC-like'})"
            )
            window_results[key] = dict(
                lo=lo, hi=hi, centre=centre, desc=desc,
                covered=covered, peak_val=mean_val,
                peak_pos=peak_pos, mean_val=mean_val,
                rel_pct=mean_rel_pct, note=note,
            )
            continue

        window_results[key] = dict(
            lo=lo, hi=hi, centre=centre, desc=desc,
            covered=covered, peak_val=peak_val,
            peak_pos=peak_pos, mean_val=mean_val,
            rel_pct=peak_val / diagnostic_max * 100,
            note=None,
        )

    sections = []
    for material, keys in MATERIAL_WINDOWS.items():
        lines = [f"  [{material}]"]
        for k in keys:
            r = window_results[k]
            if not r["covered"]:
                lines.append(f"    {r['desc']:<54s}  ← outside scan range")
                continue
            if r["peak_val"] is None:
                lines.append(f"    {r['desc']:<54s}  ← no data in window")
                continue
            if r["note"]:
                lines.append(f"    {r['desc']:<54s}  {r['note']}")
                continue
            rel   = r["rel_pct"]
            label = "STRONG" if rel > 25 else "moderate" if rel > 8 else "weak"
            lines.append(
                f"    {r['desc']:<54s}  "
                f"max={r['peak_val']:8.1f} cts  "
                f"@ {r['peak_pos']:7.1f} cm⁻¹  "
                f"({rel:5.1f}% of diagnostic_max)  [{label}]"
            )
        sections.append("\n".join(lines))

    diagnostic_table = "\n\n".join(sections)

    summary = textwrap.dedent(f"""
        SPECTRUM OVERVIEW
        -----------------
        Wavenumber range          : {x_min:.1f} – {x_max:.1f} cm⁻¹
        Intensity range           : {y_min:.1f} – {y_max:.1f} counts
        Global maximum            : {global_max:.1f} counts  (may include PL/substrate)
        Diagnostic-window maximum : {diagnostic_max:.1f} counts  (reference for % below)
        Estimated baseline (5 %)  : {baseline:.1f} counts
        Data points               : {len(x)}{pl_warning}

        DIAGNOSTIC WINDOW ANALYSIS  (literature-standard ranges)
        =========================================================
        Reported intensity = maximum value found within each window.
        Relative % is vs. diagnostic_max (strongest Raman window peak).
        Thresholds: >25 % = STRONG | 8–25 % = moderate | <8 % = weak
        TMC broad window uses mean intensity + peak/mean flatness ratio.

{diagnostic_table}
    """).strip()

    return summary



SYSTEM_PROMPT = """\
You are an expert in Raman spectroscopy of two-dimensional (2D) materials.
Classify the input spectrum as exactly one of five materials, or flag it as ambiguous.

CRITICAL RULES FOR INTERPRETING THE INPUT:
- Relative intensities are given as % of diagnostic_max, which is the strongest peak
  found across all crystalline Raman windows (Graphene, hBN, MoS2, WSe2).
- If a PL/substrate WARNING is present, the raw spectrum is dominated by an artifact.
  In that case, G and 2D bands appearing at moderate level (8-25%) can still be
  definitive for Graphene — evaluate their RATIO to each other, not their absolute level.
- Only STRONG signals (>25%) should trigger exclusion rules unless stated otherwise.

You must respond with valid JSON only. Do not include any prose, markdown, code fences,
or reasoning text outside the JSON object. Start your response directly with '{'."""

CLASSIFICATION_PROMPT = """\
================================================================
CANDIDATE MATERIALS AND RAMAN SIGNATURES
================================================================

1. GRAPHENE
   Diagnostic peaks:
   - G peak: ~1580 cm⁻¹ (E₂g mode, always present)
   - 2D peak: ~2650–2700 cm⁻¹ (second-order, always present; sharp in
     high-quality monolayer, broader/weaker in multilayer or on substrates)
   - D peak: ~1350 cm⁻¹ (absent in pristine; weak D means high quality)
   - D′ peak: ~1620 cm⁻¹ (defect-related, usually very weak)
   Key ratios (calculated from diagnostic window intensities):
   - I(2D)/I(G): monolayer > 2.0; bilayer ≈ 1.0; few-layer 0.5–1.0
   - I(D)/I(G) < 0.2 indicates good crystalline quality
   - The simultaneous presence of both G and 2D is HIGHLY SPECIFIC to graphene.
     No other candidate material produces both peaks.
   PL/substrate note: When a PL WARNING is present, G and 2D will appear at
   moderate relative intensity (8–25%) because diagnostic_max is still anchored
   to the strongest Raman peak. This does NOT weaken the graphene diagnosis —
   evaluate the G-to-2D ratio and the absence of other materials' primary peaks.
   Exclusion rule: Absence of any signal in the 2D window (2620–2780 cm⁻¹)
   rules out graphene. Presence of G alone is insufficient.

2. hBN (hexagonal boron nitride)
   Diagnostic peaks:
   - E₂g mode: ~1366 cm⁻¹ (dominant, sharp, strong; only significant peak)
   Exclusion rules:
   - STRONG peaks (>25% of diagnostic_max) near ~383, ~408, or ~2700 cm⁻¹ rule out hBN
   - Weak incidental signals (<8%) in those regions do NOT rule out hBN
   - Multiple STRONG peaks of comparable intensity rule out hBN
   - No signal in the hBN window (1355–1380 cm⁻¹) rules out hBN

3. MoS₂ (molybdenum disulfide)
   Diagnostic peaks:
   - E²₁g / E′ mode: ~383 cm⁻¹ (in-plane)
   - A₁g / A₁′ mode: ~408 cm⁻¹ (out-of-plane)
   - Both peaks must be present; separation 18–25 cm⁻¹ (less → fewer layers)
   - 2LA(M) mode: ~450 cm⁻¹ (weaker, sometimes present)
   Exclusion rules:
   - Absence of BOTH ~383 and ~408 cm⁻¹ as at least moderate signals rules out MoS₂
   - STRONG peaks dominating below ~300 cm⁻¹ rule out MoS₂
   - Presence of a STRONG 2D band (~2700 cm⁻¹) rules out pure MoS₂

4. WSe₂ (tungsten diselenide)
   Diagnostic peaks:
   - A₁g / E²₁g (near-degenerate): ~250–260 cm⁻¹ (dominant, most intense peak)
   - E₁g mode: ~308 cm⁻¹ (weaker, sometimes absent)
   - LA(M) acoustic mode: ~150–175 cm⁻¹ (may be present)
   Exclusion rules:
   - STRONG peaks (>25%) near ~383 or ~408 cm⁻¹ rule out WSe₂
   - Weak background signals (<8%) in those regions do NOT rule out WSe₂
   - STRONG peaks above 350 cm⁻¹ that exceed the ~252 cm⁻¹ peak rule out WSe₂
   - Absence of any signal in the WSe₂ A₁g/E²₁g window (238–272 cm⁻¹) rules out WSe₂

5. Ti₃C₂Tₓ (TMC — Transition Metal Carbide)
   Diagnostic features:
   - Broad, diffuse features spanning ~100–900 cm⁻¹; no sharp crystalline peaks
   - Broad bands near ~200 cm⁻¹ (C–Ti–C out-of-plane), ~400 cm⁻¹ (Ti–C stretch),
     and ~620 cm⁻¹ (Ti–O surface termination)
   - Spectrum is flat and featureless compared to TMDCs or graphene
   - TMC broad envelope: peak/mean ratio < 3 indicates flat, TMC-like spectrum
   - All TMC bands are inherently broad (FWHM >> 20 cm⁻¹); no sharp peaks expected
   Exclusion rules:
   - peak/mean ratio > 5 in the broad envelope means a sharp crystalline peak is
     present, which strongly argues against TMC
   - Any STRONG signal in the Graphene G (1545–1620), 2D (2620–2780), hBN (1355–1380),
     MoS₂ E²₁g (378–392), or WSe₂ A₁g (238–272) windows rules out TMC

================================================================
INPUT SPECTRUM DATA
================================================================
{SPECTRUM_DATA}

================================================================
CLASSIFICATION INSTRUCTIONS
================================================================

Follow these four steps before producing output:

STEP 1 — INVENTORY
  For each diagnostic window, note: position, intensity label, and % of diagnostic_max.
  If a PL WARNING is present, note it and interpret all intensities as relative to
  the strongest Raman window, not the PL artifact.

STEP 2 — COMPUTE KEY RATIOS (for Graphene)
  Calculate I(2D)/I(G) = (2D window max) / (G window max).
  Calculate I(D)/I(G)  = (D window max) / (G window max).
  These ratios are independent of PL and are reliable even when absolute
  intensities are suppressed by a dominant background.

STEP 3 — APPLY EXCLUSION RULES
  Apply each material's exclusion rules strictly. Only STRONG signals (>25%)
  trigger exclusions unless the rule explicitly states otherwise.
  Eliminated materials must not appear in the final answer.

STEP 4 — MATCH AND DECIDE
  Score remaining candidates against their expected diagnostic features.
  For Graphene: co-presence of G AND 2D at any intensity level, with a
  physically reasonable I(2D)/I(G) ratio, is sufficient for high confidence.
  Return "Ambiguous" only if two candidates match equally well, or if the
  primary diagnostic windows for all candidates are outside the scan range.

================================================================
CONFIDENCE SCALE
================================================================
- 0.9–1.0 : Primary and secondary windows match; all others clearly excluded
- 0.7–0.89: Primary windows match; secondary features absent or ambiguous
- 0.5–0.69: Primary region consistent but peak positions shifted or uncertain
- < 0.5   : Return "Ambiguous" instead

================================================================
OUTPUT FORMAT
================================================================
Return your answer strictly as valid JSON with no extra text:

{{
  "material": "<Graphene | hBN | MoS2 | WSe2 | Ti3C2Tx | Ambiguous>",
  "confidence": 0.0,
  "eliminated_candidates": {{
    "Graphene": "<reason eliminated, or not eliminated>",
    "hBN": "<reason eliminated, or not eliminated>",
    "MoS2": "<reason eliminated, or not eliminated>",
    "WSe2": "<reason eliminated, or not eliminated>",
    "Ti3C2Tx": "<reason eliminated, or not eliminated>"
  }},
  "key_evidence": [
    "specific window result supporting the decision",
    "second supporting observation"
  ],
  "reasoning": "2–4 sentence scientific justification referencing specific window positions and intensities from the input"
}}
"""




def classify_spectrum(json_path: str) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    x = data.get("x") or data.get("wavenumber") or data.get("wavenumbers")
    y = data.get("y") or data.get("intensity") or data.get("counts")

    if x is None or y is None:
        raise ValueError(
            "Could not find x/y arrays in JSON. "
            "Expected keys: 'x'/'y', 'wavenumber'/'intensity', or 'wavenumbers'/'counts'."
        )

    spectrum_id = data.get("metadata", {}).get("id") or data.get("spectrum_id", json_path)
    x_unit      = data.get("x_unit", "cm⁻¹")
    y_unit      = data.get("y_unit", "counts")

    print(f"\n{'='*60}")
    print(f"  Spectrum ID : {spectrum_id}")
    print(f"  X unit      : {x_unit}")
    print(f"  Y unit      : {y_unit}")
    print(f"  Points      : {len(x)}")
    print(f"{'='*60}\n")

    print("Analysing spectral features...")
    spectrum_summary = summarise_spectrum(x, y)
    print(spectrum_summary)
    print()

    filled_prompt = CLASSIFICATION_PROMPT.format(SPECTRUM_DATA=spectrum_summary)

    api_key = _load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)

    print("Calling Claude API for classification...\n")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": filled_prompt},
        ],
    )

    raw_text = response.content[0].text.strip()


    if "```" in raw_text:
        lines    = raw_text.splitlines()
        raw_text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()


    brace = raw_text.find("{")
    if brace > 0:
        raw_text = raw_text[brace:]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Could not parse JSON response. Raw output:\n")
        print(raw_text)
        return {"raw_response": raw_text}

    result = {
        "spectrum_id": spectrum_id,
        "source_file": os.path.basename(json_path),
        **result,
    }

    return result




def print_result(result: dict) -> None:
    if "raw_response" in result:
        return

    material   = result.get("material", "N/A")
    confidence = float(result.get("confidence", 0.0))
    eliminated = result.get("eliminated_candidates", {})
    evidence   = result.get("key_evidence", [])
    reasoning  = result.get("reasoning", "")

    bar_len = max(0, min(30, int(confidence * 30)))
    bar     = "█" * bar_len + "░" * (30 - bar_len)

    print("=" * 60)
    print("  CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"  Material   : {material}")
    print(f"  Confidence : [{bar}] {confidence:.0%}")
    print()
    print("  ELIMINATED CANDIDATES")
    print("  " + "-" * 40)
    for mat, reason in eliminated.items():
        status = "✗  eliminated" if "not eliminated" not in str(reason).lower() else "✓  retained"
        print(f"  {mat:<12s}  {status}")
        print(f"             {reason}")
        print()
    print("  KEY EVIDENCE")
    print("  " + "-" * 40)
    for ev in evidence:
        print(f"  • {ev}")
    print()
    print("  SCIENTIFIC REASONING")
    print("  " + "-" * 40)
    for line in textwrap.wrap(str(reasoning), width=56):
        print(f"  {line}")
    print("=" * 60)

    out_path = "classification_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Full JSON result saved to: {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Classify a Raman spectrum JSON file using Claude (API key loaded from .env)."
    )
    parser.add_argument(
        "spectrum_file",
        help="Path to the JSON spectrum file (must contain 'x' and 'y' arrays).",
    )
    args = parser.parse_args()

    result = classify_spectrum(args.spectrum_file)
    print_result(result)


if __name__ == "__main__":
    main()