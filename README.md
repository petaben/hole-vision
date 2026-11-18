# hole-vision

Measures the diameter of a hole in a microscope image.

## Setup

The dependencies go into a virtual environment in `.venv`, separate from your global Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in each new terminal before using the commands below.

## Web app

```bash
streamlit run vision/app.py --server.address localhost
```

opens a page in your browser (http://localhost:8501). Pick the magnification, drop in one or
more microscope images, and each one gets its diameter and the image with the measured circle,
which you can download. With several images, a table lists all diameters.

`--server.address localhost` keeps the page reachable only from your computer. Without it,
anyone on your network can open it at `http://<your-computer's-IP>:8501`; there is no login.

### Online

The app can run for free on [Streamlit Community Cloud](https://share.streamlit.io), straight
from this GitHub repository:

1. Push your changes to GitHub.
2. Sign in at https://share.streamlit.io with GitHub, click **Create app** and choose to deploy
   from a GitHub repository.
3. Pick this repository, branch `main` and main file path `vision/app.py`. Under
   **Advanced settings**, pick the newest Python version offered. Then click **Deploy**.
4. In the app's **Share** settings, choose who can open it: anyone with the link, or only people
   you invite by email.

Every push to `main` updates the online app. Images uploaded there are processed on Streamlit's
servers in the US.

## Command line

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
