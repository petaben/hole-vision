"""Web page for measuring holes: upload microscope images, get their diameters.

    streamlit run vision/app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from measure_hole import LOW_COVERAGE_WARNING, Hole, HoleNotFoundError, decode_image, draw_debug, measure_hole

# Scale of the microscope camera at each magnification (from test_data/readme.md).
PX_PER_MM = {"40x": 1016.3473, "100x": 2516.6193}
FILE_TYPES = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]


@st.cache_data(show_spinner=False, max_entries=32)
def analyse(data: bytes, px_per_mm: float) -> tuple[Hole, bytes, bytes]:
    """Measure the hole and draw it: (hole, JPEG preview, PNG for download)."""
    image = decode_image(data)
    hole = measure_hole(image, px_per_mm)
    debug = draw_debug(image, hole)
    preview = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    return hole, preview, cv2.imencode(".png", debug)[1].tobytes()


st.set_page_config(page_title="Hole measurement", layout="wide")
st.title("Hole measurement")
st.write("Upload microscope images to measure the hole in each. "
         "The green circle shows what was measured.")

magnification = st.radio(
    "Magnification",
    [*PX_PER_MM, "Other"],
    format_func=lambda m: f"{m} ({PX_PER_MM[m]} px/mm)" if m in PX_PER_MM else m,
    horizontal=True,
)
if magnification in PX_PER_MM:
    px_per_mm = PX_PER_MM[magnification]
else:
    px_per_mm = st.number_input("Scale (px/mm)", min_value=0.0001, value=None, format="%.4f",
                                placeholder="e.g. 1016.3473")

uploads = st.file_uploader("Microscope images", type=FILE_TYPES, accept_multiple_files=True)
if not uploads:
    st.stop()
if not px_per_mm:
    st.info("Enter the scale in px/mm to measure.")
    st.stop()

results = []
with st.spinner("Measuring..."):
    for upload in uploads:
        try:
            results.append((upload.name, analyse(upload.getvalue(), px_per_mm)))
        except (ValueError, HoleNotFoundError) as exc:
            results.append((upload.name, exc))

if len(results) > 1:
    st.dataframe(
        [{"image": name, "diameter (mm)": None if isinstance(r, Exception) else r[0].diameter_mm}
         for name, r in results],
        column_config={"diameter (mm)": st.column_config.NumberColumn(format="%.3f")},
        hide_index=True,
    )

for name, result in results:
    if isinstance(result, Exception):
        st.error(f"{name}: {result}")
        continue
    hole, preview, png = result
    with st.expander(f"{name}: {hole.diameter_mm:.3f} mm", expanded=len(results) == 1):
        left, right = st.columns([3, 1])
        left.image(preview, width="stretch")
        right.metric("Diameter", f"{hole.diameter_mm:.3f} mm")
        right.caption(f"Rim found on {hole.coverage:.0%} of the circumference. "
                      f"Centre ({hole.center_x:.0f}, {hole.center_y:.0f}) px, radius {hole.radius_px:.1f} px.")
        if hole.coverage < LOW_COVERAGE_WARNING:
            right.warning("Only part of the rim was found. Check that the circle is right.")
        right.download_button("Download image", png, file_name=f"{Path(name).stem}_debug.png",
                              mime="image/png", on_click="ignore")
