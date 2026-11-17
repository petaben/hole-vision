#!/usr/bin/env python3
"""Measure the diameter of a hole in a microscope image.

    python vision/measure_hole.py IMAGE PX_PER_MM [-d] [-o PATH]

Prints the hole diameter in millimetres. With -d it also writes the input image with the
detected hole circled (default: <image>_debug.png next to the input).

The hole is expected to show up as a bright, roughly circular area (light shining through
it) bounded by a darker rim or background. Detection runs in two stages:

1. Coarse: Hough circle transforms and a bright-blob search on a downscaled copy propose
   candidate circles.
2. Fine: for each candidate, rays are cast from its centre and the bright-to-dark edge is
   located on every ray with sub-pixel precision (steepest intensity drop). A circle is
   fitted to those edge points while rejecting outliers, so burrs, chips and debris on the
   rim do not skew the result. The candidate whose circle is backed by the strongest and
   most complete edge wins.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EDGE_BLUR_SIGMA = 1.5  # px, noise suppression before locating edges
EDGE_SNR = 5.0  # an edge must be this many times steeper than the image noise
NOISE_FLOOR = 0.25  # grey levels per px; flat 8-bit backgrounds would otherwise read as noise-free
COARSE_SIZE = 640  # px, longest side of the image used for the coarse search
MIN_COVERAGE = 0.3  # fraction of the rim that must be found to accept a circle
LOW_COVERAGE_WARNING = 0.6


@dataclass
class Hole:
    """A detected hole. Pixel values refer to the full-resolution input image."""

    center_x: float
    center_y: float
    radius_px: float
    diameter_mm: float
    coverage: float  # fraction of the rim where an edge was found that fits the circle
    rim_deviation_px: float  # robust standard deviation of the rim around the circle


@dataclass
class _Fit:
    cx: float
    cy: float
    r: float
    coverage: float
    support: float  # mean edge strength along the whole circumference
    deviation: float


class HoleNotFoundError(RuntimeError):
    pass


def load_image(path: str | Path) -> np.ndarray:
    # Microscope TIFFs carry vendor tags that make libtiff warn on every read.
    log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


def measure_hole(image: np.ndarray, px_per_mm: float) -> Hole:
    """Find the hole in a BGR or grayscale image and measure its diameter."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    gray = _to_gray(image)
    smooth = cv2.GaussianBlur(gray, (0, 0), EDGE_BLUR_SIGMA)
    noise = 1.4826 * float(np.median(np.abs(np.diff(smooth, axis=1))))
    min_strength = EDGE_SNR * max(noise, NOISE_FLOOR)

    best = None
    for cx, cy, r in _coarse_candidates(gray):
        fit = _refine(smooth, cx, cy, r, min_strength)
        if fit and fit.coverage >= MIN_COVERAGE and (best is None or fit.support > best.support):
            best = fit
    if best is None:
        raise HoleNotFoundError("no hole found in the image")

    return Hole(
        center_x=best.cx,
        center_y=best.cy,
        radius_px=best.r,
        diameter_mm=2 * best.r / px_per_mm,
        coverage=best.coverage,
        rim_deviation_px=best.deviation,
    )


def draw_debug(image: np.ndarray, hole: Hole) -> np.ndarray:
    """Return a copy of the image with the detected hole circled and labelled."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = out.shape[:2]
    color = (0, 255, 0)
    thickness = max(1, round(max(h, w) / 1000))

    shift = 4  # draw with sub-pixel precision
    center = (round(hole.center_x * 2**shift), round(hole.center_y * 2**shift))
    cv2.circle(out, center, round(hole.radius_px * 2**shift), color, thickness, cv2.LINE_AA, shift)
    marker = max(10, round(0.06 * hole.radius_px))
    cv2.drawMarker(out, (round(hole.center_x), round(hole.center_y)), color, cv2.MARKER_CROSS, marker, thickness)

    text = f"D = {hole.diameter_mm:.3f} mm"
    scale = max(h, w) / 1600
    text_thickness = max(1, round(2 * scale))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
    pad = round(10 * scale)
    x0, y0 = pad, h - pad - th - baseline - 2 * pad
    cv2.rectangle(out, (x0, y0), (x0 + tw + 2 * pad, h - pad), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, text, (x0 + pad, h - 2 * pad - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                text_thickness, cv2.LINE_AA)
    return out


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(_remove_annotations(image), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _remove_annotations(bgr: np.ndarray) -> np.ndarray:
    """Paint over red overlay lines and labels burned in by the microscope software."""
    b, g, r = (c.astype(np.int16) for c in cv2.split(bgr))
    mask = ((r > 100) & (r - np.maximum(g, b) > 80)).astype(np.uint8)
    # Overlays are thin lines and text; a large red area is image content, leave it alone.
    if not mask.any() or mask.mean() > 0.02:
        return bgr
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def _coarse_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Rough (cx, cy, r) guesses in full-resolution pixels."""
    h, w = gray.shape
    factor = max(1, round(max(h, w) / COARSE_SIZE))
    small = cv2.resize(gray, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    # Stretch contrast for the Hough edge thresholds. Keep the brightest pixels unclipped:
    # a small hole may be all of them.
    lo, hi = np.percentile(small, 0.5), small.max()
    small = np.clip((small - lo) * (255 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    sh, sw = small.shape
    min_r, max_r = 6, int(0.6 * min(sh, sw))

    found = []
    for method, param2 in ((cv2.HOUGH_GRADIENT_ALT, 0.7), (cv2.HOUGH_GRADIENT, 25)):
        circles = cv2.HoughCircles(small, method, dp=1, minDist=10, param1=60, param2=param2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            found.extend(tuple(c) for c in circles[0][:5])

    # Fallback for when Hough finds nothing useful: the largest bright blobs.
    _, bright = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(bright)
    areas = stats[1:, cv2.CC_STAT_AREA]
    for i in np.argsort(areas)[::-1][:3] + 1:
        if stats[i, cv2.CC_STAT_AREA] >= np.pi * min_r**2:
            found.append((*centroids[i], np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)))

    sx, sy = w / sw, h / sh
    candidates: list[tuple[float, float, float]] = []
    for x, y, r in found:
        c = ((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5, r * (sx + sy) / 2)
        # Several guesses usually describe the same circle; refining one of them is enough.
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) > 0.05 * o[2] or abs(c[2] - o[2]) > 0.05 * o[2]
               for o in candidates):
            candidates.append(c)
    return candidates


def _refine(smooth: np.ndarray, cx: float, cy: float, r: float, min_strength: float) -> _Fit | None:
    """Fit a circle to the bright-to-dark edge near the circle (cx, cy, r)."""
    # Wide search bands first to reach the edge from a rough guess, then narrow bands
    # around the improved circle so edges further away cannot interfere.
    for band_fraction, max_rays in ((0.5, 360), (0.15, 720), (0.03, 3600), (0.03, 3600)):
        band = max(band_fraction * r, 8.0)
        n_rays = int(np.clip(2 * np.pi * r, 90, max_rays))
        x, y, strength = _edge_points(smooth, cx, cy, r, band, n_rays)
        # Rays that only found noise, or a much weaker edge than most rays, do not count.
        strong = strength > max(min_strength, 0.3 * np.median(strength))
        if strong.sum() < max(10, MIN_COVERAGE * n_rays):
            return None
        cx, cy, r, inliers, deviation = _fit_circle_robust(x[strong], y[strong])
        if not (np.isfinite(r) and r > 1):
            return None
    return _Fit(
        cx=cx,
        cy=cy,
        r=r,
        coverage=inliers.sum() / n_rays,
        support=strength[strong][inliers].sum() / n_rays,
        deviation=deviation,
    )


def _edge_points(smooth: np.ndarray, cx: float, cy: float, r: float, band: float, n_rays: int,
                 step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """On each of n_rays rays, find where the intensity drops fastest within r +- band.

    Returns the edge points (x, y) and the steepness of the drop at each of them.
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    radii = np.arange(max(r - band, 0.3 * r), r + band, step)
    map_x = (cx + np.outer(cos, radii)).astype(np.float32)
    map_y = (cy + np.outer(sin, radii)).astype(np.float32)
    # Outside the image the border pixel is repeated: flat, so no false edges.
    profiles = cv2.remap(smooth, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    slope = np.gradient(profiles, step, axis=1)  # intensity change per pixel outwards
    k = np.argmin(slope[:, 1:-1], axis=1) + 1  # steepest bright-to-dark drop
    rows = np.arange(n_rays)
    before, at, after = slope[rows, k - 1], slope[rows, k], slope[rows, k + 1]
    # Sub-pixel position: vertex of the parabola through the three slope samples.
    curvature = before - 2 * at + after
    offset = np.divide(before - after, 2 * curvature, out=np.zeros_like(curvature),
                       where=np.abs(curvature) > 1e-6)
    edge_r = radii[k] + np.clip(offset, -0.5, 0.5) * step
    return cx + cos * edge_r, cy + sin * edge_r, -at


def _fit_circle_robust(x: np.ndarray, y: np.ndarray, k: float = 3.0, min_tolerance: float = 2.0):
    """Circle fit that iteratively drops points further than k sigma (MAD) from the circle."""
    inliers = np.ones(len(x), dtype=bool)
    for _ in range(30):
        cx, cy, r = _fit_circle(x[inliers], y[inliers])
        residual = np.hypot(x - cx, y - cy) - r
        res_in = residual[inliers]
        sigma = 1.4826 * np.median(np.abs(res_in - np.median(res_in)))
        updated = np.abs(residual) <= max(k * sigma, min_tolerance)
        if updated.sum() < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated
    return cx, cy, r, inliers, sigma


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle: algebraic (Kasa) fit refined by Gauss-Newton on the distances."""
    a = np.column_stack([x, y, np.ones_like(x)])
    (p, q, s), *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = p / 2, q / 2
    r = np.sqrt(s + cx * cx + cy * cy)
    for _ in range(10):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy)
        jacobian = np.column_stack([dx / d, dy / d, np.ones_like(d)])
        delta, *_ = np.linalg.lstsq(jacobian, d - r, rcond=None)
        cx, cy, r = cx + delta[0], cy + delta[1], r + delta[2]
        if np.abs(delta).max() < 1e-4:
            break
    return float(cx), float(cy), float(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the diameter of a hole in a microscope image.")
    parser.add_argument("image", type=Path, help="path to the microscope image (TIFF, PNG, JPEG, ...)")
    parser.add_argument("px_per_mm", type=float,
                        help="image scale in pixels per millimetre, e.g. 1016.3473 at 40x")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="also write a copy of the image with the detected hole circled")
    parser.add_argument("-o", "--debug-output", type=Path, metavar="PATH",
                        help="where to write the debug image (default: <image>_debug.png next to "
                             "the input); implies -d")
    args = parser.parse_args(argv)
    if args.px_per_mm <= 0:
        parser.error("px_per_mm must be positive")

    try:
        image = load_image(args.image)
        hole = measure_hole(image, args.px_per_mm)
    except (FileNotFoundError, HoleNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if hole.coverage < LOW_COVERAGE_WARNING:
        print(f"warning: an edge was found on only {hole.coverage:.0%} of the rim; "
              "check the result with -d", file=sys.stderr)
    print(f"{hole.diameter_mm:.3f}")

    if args.debug or args.debug_output:
        path = args.debug_output or args.image.with_name(f"{args.image.stem}_debug.png")
        try:
            written = cv2.imwrite(str(path), draw_debug(image, hole))
        except cv2.error:  # e.g. unknown file extension
            written = False
        if not written:
            print(f"error: cannot write debug image: {path}", file=sys.stderr)
            return 1
        print(f"debug image: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
