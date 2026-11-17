# hole_vision

Measures the diameter of a hole in a microscope image.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python vision/measure_hole.py test_data/1-197.tif 1016.3473
```

prints the hole diameter in mm, e.g. `1.193`.

| Argument | |
| --- | --- |
| `image` | path to the image (TIFF, PNG, JPEG, ...) |
| `px_per_mm` | image scale: `1016.3473` at 40x, `2516.6193` at 100x |
| `-d` | also write the image with the detected hole circled, as `<image>_debug.png` next to the input |
| `-o PATH` | write the debug image to `PATH` instead (implies `-d`) |

Only the diameter goes to stdout; warnings and the debug image path go to stderr. The exit
code is 1 if the image cannot be read or no hole is found.

From Python, `measure_hole(load_image(path), px_per_mm)` in `vision/measure_hole.py` returns
the diameter together with the centre and radius in pixels.

## How it works

The hole is the bright, roughly circular area where light shines through, bounded by a darker
rim or background.

1. Red overlay lines and labels burned in by the microscope software are painted out.
2. Hough circle transforms on a downscaled copy of the image propose candidate circles.
3. For each candidate, rays are cast from its centre and the edge on each ray is placed where
   the intensity drops fastest, with sub-pixel precision.
4. A circle is fitted to those edge points, repeatedly dropping points that lie far off the
   circle, so burrs, chips and debris on the rim are ignored.
5. The candidate backed by the strongest, most complete edge wins.

The reported diameter is that of the fitted circle: the rim's mean diameter with burrs
ignored, not the narrowest gap between burrs.

Assumptions: the hole appears brighter than its rim, and most of its rim is inside the image.
If less than 60% of the rim is found, a warning is printed.

## Accuracy

```bash
python vision/evaluate.py
```

measures every image in `test_data/` and compares it with the reference diameter in its file
name (`1-197.tif` = 1.197 mm, measured by hand in the microscope software). The worst error
is 0.012 mm (12 px) at 40x and 0.004 mm (9 px) at 100x, with no systematic bias (mean error
+0.5 px). The remaining scatter is mostly where along the blurred edge the reference line was
placed by hand.
