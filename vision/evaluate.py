#!/usr/bin/env python3
"""Compare measure_hole against the reference measurements of the test images.

    python vision/evaluate.py [TEST_DATA_DIR]

Test image names encode the diameter measured with the microscope software: "1-197.tif" is
1.197 mm. A "_100x" suffix marks 100x magnification, all others are 40x. Exits with status 1
if any image is off by more than TOLERANCE_PX.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from measure_hole import HoleNotFoundError, load_image, measure_hole

PX_PER_MM = {40: 1016.3473, 100: 2516.6193}  # from test_data/readme.md
TOLERANCE_PX = 15  # reference diameters were set by hand on screen
NAME = re.compile(r"(\d+)-(\d+)(?:_(\d+)x)?")


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "test_data"
    images = sorted(p for p in data_dir.iterdir() if NAME.fullmatch(p.stem) and p.suffix.lower() in (".tif", ".tiff"))
    if not images:
        print(f"no test images found in {data_dir}", file=sys.stderr)
        return 1

    print(f"{'image':<16}{'ref mm':>8}{'meas mm':>9}{'err um':>8}{'err px':>8}{'rim':>6}{'time':>7}")
    errors_px = []
    failed = []
    for path in images:
        whole, frac, mag = NAME.fullmatch(path.stem).groups()
        reference = float(f"{whole}.{frac}")
        px_per_mm = PX_PER_MM[int(mag or 40)]
        start = time.perf_counter()
        try:
            hole = measure_hole(load_image(path), px_per_mm)
        except HoleNotFoundError:
            print(f"{path.name:<16}{reference:>8.3f}{'not found':>9}")
            failed.append(path.name)
            continue
        elapsed = time.perf_counter() - start
        err_mm = hole.diameter_mm - reference
        err_px = err_mm * px_per_mm
        errors_px.append(err_px)
        flag = "" if abs(err_px) <= TOLERANCE_PX else "  <-- off"
        if flag:
            failed.append(path.name)
        print(f"{path.name:<16}{reference:>8.3f}{hole.diameter_mm:>9.3f}{1000 * err_mm:>+8.1f}{err_px:>+8.1f}"
              f"{hole.coverage:>6.0%}{elapsed:>6.2f}s{flag}")

    if errors_px:
        abs_err = [abs(e) for e in errors_px]
        print(f"\nmean error {sum(errors_px) / len(errors_px):+.1f} px, "
              f"mean |error| {sum(abs_err) / len(abs_err):.1f} px, max |error| {max(abs_err):.1f} px "
              f"(tolerance {TOLERANCE_PX} px)")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
