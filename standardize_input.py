import argparse
import json
import hashlib
import os
import re
import numpy as np


def generate_spectrum_id(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash[:16]


def parse_raman_file(filepath):

    xs = []
    ys = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            line = re.sub(r"[;\t]+", " ", line)

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                x = float(parts[0].replace(",", "."))
                y = float(parts[1].replace(",", "."))

                xs.append(x)
                ys.append(y)

            except ValueError:
                continue

    if len(xs) < 10:
        raise ValueError("Not enough numeric data found.")

    x = np.array(xs)
    y = np.array(ys)

    if np.any(np.diff(x) < 0):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    return x.tolist(), y.tolist()


def create_json_structure(filepath):

    x, y = parse_raman_file(filepath)

    spectrum_id = generate_spectrum_id(filepath)

    data = {
        "spectrum_id": spectrum_id,
        "x_unit": "cm^-1",
        "y_unit": "counts",
        "x": x,
        "y": y,
        "metadata": {
            "laser_nm": None,
            "power_mw": None,
            "integration_s": None,
            "objective": None,
            "substrate": None,
            "temperature_K": None,
            "map": None
        }
    }

    return data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    data = create_json_structure(args.input_file)

    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(args.input_file)[0]
        output_file = base + ".json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Standardized JSON written to: {output_file}")


if __name__ == "__main__":
    main()