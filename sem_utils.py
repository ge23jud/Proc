"""Pure data/image-processing logic for SEM nanowire measurement — no PyQt
imports, mirroring io_utils.py's UI-free convention. Python port of the
"smarter SEM" MATLAB toolset (readSEM.m / edgeDetect.m / edgeDetect_semiauto.m
/ fit_rectangle.m / click_on_image.m).
"""
import re

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

_SCALE_UNITS = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "mm": 1e-3}
_PIXEL_SIZE_RE = re.compile(rb"Pixel Size\s*=\s*([0-9.]+)\s*([a-zA-Z\xb5]+)")


def read_sem_image(path):
    """Load a SEM TIFF as a grayscale array and parse its embedded pixel
    scale from the instrument's ASCII metadata block (regex-scanned across
    the whole file rather than assuming a fixed line offset, unlike
    readSEM.m's `headerLines=41`, since that offset is brittle across
    acquisition settings).

    Returns (image: np.ndarray uint8 (H, W), scale_m_per_px: float | None).
    scale is None if no parseable "Pixel Size = <value> <unit>" field is found.
    """
    image = np.array(Image.open(path).convert("L"))
    scale = None
    with open(path, "rb") as fh:
        raw = fh.read()
    m = _PIXEL_SIZE_RE.search(raw)
    if m:
        value = float(m.group(1))
        unit = m.group(2).decode("latin-1").lower()
        factor = _SCALE_UNITS.get(unit)
        if factor is not None:
            scale = value * factor
    return image, scale


def _min_area_rect(points_xy):
    """Fit a minimum-area rotated rectangle to a boundary point cloud
    (Nx2, x/y = col/row pixel coords), the Python/cv2 replacement for
    fit_rectangle.m's PCA/extreme-point approximation.

    Returns dict: width_px (short side), height_px (long side),
    angle_deg (long-axis tilt from vertical, sign ambiguous but magnitude
    meaningful — matches fit_rectangle.m's convention of rejecting wires
    tilted more than a few degrees off vertical), center (x, y),
    box_points (4x2, cv2.boxPoints order).
    """
    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    edge0 = box[1] - box[0]
    edge1 = box[2] - box[1]
    len0, len1 = float(np.linalg.norm(edge0)), float(np.linalg.norm(edge1))
    if len0 >= len1:
        long_vec, length, diameter = edge0, len0, len1
    else:
        long_vec, length, diameter = edge1, len1, len0
    # atan2(dx, dy): 0 when the long axis is vertical, +/-90 when horizontal.
    # Wrapped to (-90, 90] so a vector pointing "up" vs "down" along the same
    # line reports the same tilt magnitude.
    angle = np.degrees(np.arctan2(long_vec[0], long_vec[1]))
    angle = ((angle + 90) % 180) - 90
    return {
        "width_px": diameter,
        "height_px": length,
        "angle_deg": float(angle),
        "center": (float(rect[0][0]), float(rect[0][1])),
        "box_points": box,
    }


def _binarize(image, method="global", threshold=120):
    if method == "adaptive":
        bw = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=51, C=-10,
        )
    else:
        _, bw = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return bw.astype(bool)


def detect_wires(image, scale_m_per_px, threshold=120, method="global",
                  edge_margin=4, min_boundary_points=20, max_angle_deg=5.0,
                  skip_ids=()):
    """Whole-image blob detection + rectangle fit, port of edgeDetect.m.

    Binarizes, fills interior holes, traces external contours, then fits
    each one to a minimum-area rectangle and applies the same rejection
    rules as the MATLAB tool (touches image edge, too few boundary points,
    explicitly skipped index, tilted more than max_angle_deg from vertical).

    Returns a list of dicts (one per contour found, in contour order — the
    numbering used for `skip_ids`), each with keys:
      id, accepted, reject_reason, width_m, height_m, angle_deg,
      aspect_ratio, center_px, box_points_px, contour_px
    Rejected entries have width_m/height_m/aspect_ratio/center_px/box_points_px
    left as None (angle_deg is still filled in for angle-rejected entries,
    so the UI can show why a blob was skipped).
    """
    bw = _binarize(image, method=method, threshold=threshold)
    filled = ndimage.binary_fill_holes(bw)
    mask_u8 = filled.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows, cols = image.shape

    results = []
    for i, contour in enumerate(contours, start=1):
        pts = contour.reshape(-1, 2)
        reject_reason = None
        if i in skip_ids:
            reject_reason = "skipped"
        elif len(pts) < min_boundary_points:
            reject_reason = "too small"
        elif ((pts[:, 0] < edge_margin).any() or (pts[:, 0] > cols - edge_margin).any()
                or (pts[:, 1] < edge_margin).any() or (pts[:, 1] > rows - edge_margin).any()):
            reject_reason = "touches edge"

        entry = {
            "id": i, "contour_px": pts, "accepted": False,
            "width_m": None, "height_m": None, "angle_deg": None,
            "aspect_ratio": None, "center_px": None, "box_points_px": None,
            "reject_reason": reject_reason,
        }

        if reject_reason is None:
            fit = _min_area_rect(pts)
            if abs(fit["angle_deg"]) > max_angle_deg:
                entry["reject_reason"] = "tilted"
                entry["angle_deg"] = fit["angle_deg"]
            else:
                width_m = fit["width_px"] * scale_m_per_px
                height_m = fit["height_px"] * scale_m_per_px
                entry.update({
                    "accepted": True,
                    "width_m": width_m,
                    "height_m": height_m,
                    "angle_deg": fit["angle_deg"],
                    "aspect_ratio": (height_m / width_m) if width_m else None,
                    "center_px": fit["center"],
                    "box_points_px": fit["box_points"],
                })
        results.append(entry)
    return results


def crop_around(image, scale_m_per_px, center_px, size_nm):
    """Crop `image` to a physical (width_nm, length_nm) window centered at
    center_px = (x, y), clamped to image bounds. Port of smarterSEM.m's
    get_start_end_indices + crop_vector.

    Returns (crop: np.ndarray, offset: (x0, y0)) — offset is the crop's
    top-left corner in the original image's pixel coordinates, needed to
    map detections found in the crop back onto the full image.
    """
    width_nm, length_nm = size_nm
    half_w_px = (width_nm * 1e-9 / scale_m_per_px) / 2.0
    half_h_px = (length_nm * 1e-9 / scale_m_per_px) / 2.0
    rows, cols = image.shape
    cx, cy = center_px
    x0 = int(max(0, round(cx - half_w_px)))
    x1 = int(min(cols, round(cx + half_w_px)))
    y0 = int(max(0, round(cy - half_h_px)))
    y1 = int(min(rows, round(cy + half_h_px)))
    return image[y0:y1, x0:x1], (x0, y0)


def detect_best_wire_in_crop(crop, scale_m_per_px, min_height_m=10e-9, **detect_kwargs):
    """Semi-auto single-crop detection, port of edgeDetect_semiauto.m: runs
    the same detect_wires() pipeline (adaptive threshold, since crops are
    small and locally lit) and keeps only the tallest accepted blob.

    Returns the winning detection dict (see detect_wires), or None if no
    blob was accepted or the tallest one is shorter than min_height_m —
    in either case the caller should fall back to a manual 2-click
    measurement on the crop.
    """
    detect_kwargs.setdefault("method", "adaptive")
    detections = detect_wires(crop, scale_m_per_px, **detect_kwargs)
    accepted = [d for d in detections if d["accepted"]]
    if not accepted:
        return None
    best = max(accepted, key=lambda d: d["height_m"])
    if best["height_m"] < min_height_m:
        return None
    return best


def measure_two_points(p1, p2, scale_m_per_px):
    """Manual 2-click distance measurement, port of click_on_image.m.
    p1, p2: (x, y) pixel coordinates. Returns dx_m, dy_m, d_m."""
    dx_px = abs(p1[0] - p2[0])
    dy_px = abs(p1[1] - p2[1])
    d_px = float(np.hypot(dx_px, dy_px))
    return {
        "dx_m": dx_px * scale_m_per_px,
        "dy_m": dy_px * scale_m_per_px,
        "d_m": d_px * scale_m_per_px,
    }
