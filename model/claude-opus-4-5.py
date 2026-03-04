import json
import sys
import re
import os
from pathlib import Path
from anthropic import Anthropic


def load_dotenv(path=".env"):
    """Load a .env file into os.environ without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_dotenv()

MODEL = "claude-opus-4-5-20251101"
client = Anthropic() 


SYSTEM_PROMPT = """You are an expert Raman spectroscopist with comprehensive knowledge of 2D materials, transition metal dichalcogenides (TMDs), layered van der Waals materials, carbon allotropes, oxides, and common substrate and contaminant signals.

You will receive a JSON object containing a fully preprocessed Raman spectrum. The fields are:

- x_range_cm1: the spectral window covered
- qc: signal quality metrics (SNR, peak-to-mean ratio, total peak count)
- region_fractions: fraction of integrated denoised intensity in six diagnostic windows:
    100–300, 300–500, 500–900, 1300–1400, 1550–1620, 2650–2750 cm⁻¹
- peaks: all detected peaks, each with:
    cm1 (position), rel_height (height normalised to spectrum max),
    prominence (above local baseline), fwhm_cm1, left_half / right_half (half-max bounds)
- signals.y_denoised: the full baseline-corrected, denoised intensity trace

YOUR REASONING PROCESS — follow this order:

1. DOMINANT FEATURES
   Identify the 3–5 highest-prominence peaks. Note their exact positions, relative heights,
   and FWHM values. These carry the most diagnostic weight.

2. SECONDARY FEATURES
   Note any weaker but distinctive peaks that constrain the identification.
   Pay attention to peak separations (e.g. doublet splittings), overtone/combination positions,
   and the presence or absence of features in the D, G, 2D carbon windows.

3. SPECTRAL WEIGHT DISTRIBUTION
   Use region_fractions to confirm where the bulk of spectral intensity sits.
   Cross-check against candidate materials.

4. CANDIDATE EVALUATION
   List the most plausible materials. For each, state explicitly what matches and what
   does not. Rule out candidates where the evidence is inconsistent.

5. DECISION
   Commit to the best identification. Assign a confidence score (0.0–1.0) that honestly
   reflects the ambiguity or uniqueness of the fingerprint.

IMPORTANT CONSTRAINTS:
- Reason only from the spectral data. Ignore file names, IDs, or any metadata.
- Do not assume any material class is more likely than another a priori.
- If a substrate signal (e.g. Si at ~520 cm⁻¹) is present alongside weaker signals,
  identify both — report the sample material as the thin-film or flake, not just the substrate.
- Be precise: cite peak positions and values directly from the data, not generic literature ranges.

OUTPUT FORMAT — respond with a JSON object and nothing else. No preamble, no markdown fences.

{
  "material": "<your best identification, including substrate if relevant>",
  "confidence": <0.0 to 1.0>,
  "eliminated_candidates": {
    "<candidate>": "<why ruled out, citing specific spectral evidence>"
  },
  "key_evidence": [
    "<each point cites a specific peak position, separation, FWHM, or region fraction>"
  ],
  "reasoning": "<full step-by-step narrative following the five-stage process above>"
}"""


def build_user_message(spectrum: dict) -> str:
    # Send everything except the raw original signal — denoised is sufficient
    payload = {k: v for k, v in spectrum.items() if k != "signals"}
    payload["signals"] = {
        "y_denoised": spectrum.get("signals", {}).get("y_denoised", [])
    }
    return (
        "Identify the material in this Raman spectrum. "
        "Follow the five-stage reasoning process in your instructions.\n\n"
        + json.dumps(payload, indent=2)
    )


def classify(spectrum: dict) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(spectrum)}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify.py <preprocessed_spectrum.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"Error: file not found — {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        spectrum = json.load(f)

    output_path = f"{MODEL}_analysis_sample_1.json"

    print(f"Spectrum ID : {spectrum.get('spectrum_id', 'unknown')}")
    print(f"Model       : {MODEL}")
    print("Classifying...")

    result = classify(spectrum)

    output = {
        "model": MODEL,
        "spectrum_id": spectrum.get("spectrum_id", "unknown"),
        "source_file": input_path,
        **result,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {output_path}")
    print(f"Material   : {result.get('material', '—')}")
    print(f"Confidence : {result.get('confidence', '—')}")
    print("\nKey evidence:")
    for e in result.get("key_evidence", []):
        print(f"  · {e}")


if __name__ == "__main__":
    main()