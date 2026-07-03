import os
import re
import sys
import numpy as np
import h5py

# ------------------------------------------------------------------
# Import stitching logic from pl.py (sibling directory)
# ------------------------------------------------------------------
_PL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PL Helper")
)
if _PL_DIR not in sys.path:
    sys.path.insert(0, _PL_DIR)

from pl import _HC_EV_NM


def _parse_header_center_disp(hdr):
    """Extract (center_nm, disp_nm) from raw_header_lines[5] and [6].

    Returns (None, None) if the lines are absent or unparseable.
    """
    raw = hdr.get("raw_header_lines", [])
    center_nm = disp_nm = None
    if len(raw) > 5:
        m = re.search(r"([\d.]+)\s*nm", raw[5])
        if m:
            center_nm = float(m.group(1))
    if len(raw) > 6:
        m = re.search(r"([\d.]+)\s*nm", raw[6])
        if m:
            disp_nm = float(m.group(1))
    return center_nm, disp_nm


def _write_origin_file(output_path, wl_out, counts_out, header_meta, powers_W):
    """Write a power-series spectrum in LabControl .origin format.

    The header is reconstructed from header_meta (raw_header_lines + computed
    center/dispersion); only the fields that exist in the .origin format are
    written — no HDF5-specific metadata.
    """
    wl_min, wl_max = float(wl_out.min()), float(wl_out.max())
    center_nm      = (wl_min + wl_max) / 2.0
    center_eV      = _HC_EV_NM / center_nm
    disp_nm        = wl_max - wl_min
    disp_eV        = abs(_HC_EV_NM / wl_min - _HC_EV_NM / wl_max)
    int_time_str   = header_meta.get("int_time_str", "0.500")
    raw            = header_meta.get("raw_header_lines", [])

    # Parse HWP positions from header_meta
    hwp_raw  = header_meta.get("hwp_raw", [])
    hwp_vals = []
    if hwp_raw:
        first_part = hwp_raw[0].partition("\t")[2] if "\t" in hwp_raw[0] else ""
        rest_parts = " ".join(l.strip() for l in hwp_raw[1:])
        array_str  = (first_part + " " + rest_parts).replace("[", "").replace("]", "")
        try:
            hwp_vals = [float(v) for v in re.split(r"\s+", array_str.strip()) if v]
        except ValueError:
            hwp_vals = []

    def _raw_line(idx, fallback=""):
        """Return raw header line idx, stripped of newline."""
        if idx < len(raw):
            return raw[idx].rstrip("\r\n")
        return fallback

    n_p = len(powers_W)

    with open(output_path, "w", encoding="utf-8", newline="\r\n") as fh:
        # ── 9-line header ─────────────────────────────────────────
        fh.write(_raw_line(0, "Date:\t") + "\n")
        fh.write(_raw_line(1, "Measurement type:\tPowerseries") + "\n")
        fh.write(_raw_line(2, "Temperature: \t0.000 K") + "\n")
        fh.write(_raw_line(3, f"Integration time:\t{int_time_str} s") + "\n")
        # Line 4: use first power value (matches original single-file convention)
        fh.write(f"Excitation power:\t{powers_W[0]*1e3:.4g} mW\n")
        # Lines 5–6: update with actual center/dispersion
        fh.write(f"Center wavelength\t{center_nm:.3f} nm / {center_eV:.3f} eV\n")
        fh.write(
            f"Dispersion window:\t{disp_nm:.3f} nm / {disp_eV:.3f} eV\n"
        )
        fh.write(_raw_line(7, "Entrance slit width:\t0.500 mm") + "\n")
        fh.write(_raw_line(8, "Exit slit width:\t0.000 mm") + "\n")

        # ── Column headers ────────────────────────────────────────
        fh.write(" \t\n")
        fh.write("Wavelength \tPowerspectrum \n")
        fh.write(f"(nm)\t(Counts/{int_time_str}s)\n")

        # ── Power / HWP rows ──────────────────────────────────────
        power_row = "Excitation power (W)\t" + "\t".join(f"{p:.7g}" for p in powers_W)
        fh.write(power_row + "\n")

        if hwp_vals and len(hwp_vals) >= n_p:
            hwp_str = " ".join(f"{v:.1f}." for v in hwp_vals[:n_p])
            fh.write(f"Power HWP Position (°)\t{hwp_str}\n")
        else:
            fh.write("Power HWP Position (°)\t\n")

        # ── Spectral data ─────────────────────────────────────────
        for i, wl in enumerate(wl_out):
            row_counts = "\t".join(str(int(round(counts_out[i, p])))
                                   for p in range(n_p))
            fh.write(f"{wl:.9f}\t{row_counts}\n")


def _write_h5_file(output_path, wl_out, counts_diff, counts_raw, header_meta,
                   powers_W, stitched=False, source_datasets=None,
                   dark_by_label=None, spot_diameter_um=None, rep_rate_mhz=None,
                   power_cal=None):
    """Write a power-series spectrum to HDF5, preserving all .origin metadata.

    Structure
    ---------
    /Energy                     float64 (n_wl,)           — eV
    /SpectraDiff                float64 (n_wl, n_powers)  — dark-subtracted, min→1
    /Spectra_raw                float64 (n_wl, n_powers)  — stitched, no dark sub
    /Power_uncalibrated         float64 (n_powers,)       — W
    /hwp_positions              float64 (n_powers,)       — degrees (if available)
    /darkspec                   float64 (n_wl,)           — mean dark (non-stitched only)
    /source_spectra/<label>/    group, one per input file (stitched files only)
        wavelength_nm           float64 (n_wl_i,)
        counts                  float64 (n_wl_i, n_powers)  — raw (not dark-subtracted)
        darkspec                float64 (n_wl_i,)            — mean dark (if available)
    Root attrs: format_version, stitched, date_str, int_time_str,
                center_nm, center_eV, dispersion_nm, dispersion_eV,
                spot_diameter_um (optional), rep_rate_mhz (optional),
                header_line_0 … header_line_8
    """
    # Accept wl_out in nm (> 50) or eV (< 50); always save as eV.
    if float(wl_out.min()) < 50.0:          # already eV
        ev_out    = wl_out.copy()
        wl_min_nm = float(_HC_EV_NM / wl_out.max())
        wl_max_nm = float(_HC_EV_NM / wl_out.min())
    else:                                   # nm → eV
        ev_out    = _HC_EV_NM / wl_out
        wl_min_nm = float(wl_out.min())
        wl_max_nm = float(wl_out.max())
    center_nm = (wl_min_nm + wl_max_nm) / 2.0
    center_eV = _HC_EV_NM / center_nm
    disp_nm   = wl_max_nm - wl_min_nm
    disp_eV   = abs(_HC_EV_NM / wl_min_nm - _HC_EV_NM / wl_max_nm)
    int_time_str    = header_meta.get("int_time_str", "0.500")

    hwp_raw   = header_meta.get("hwp_raw", [])
    hwp_label = ""
    hwp_vals  = []
    if hwp_raw:
        hwp_label  = hwp_raw[0].partition("\t")[0]
        first_part = hwp_raw[0].partition("\t")[2] if "\t" in hwp_raw[0] else ""
        rest_parts = " ".join(l.strip() for l in hwp_raw[1:])
        array_str  = (first_part + " " + rest_parts).replace("[", "").replace("]", "")
        hwp_vals   = [v for v in re.split(r"\s+", array_str.strip()) if v]

    raw      = header_meta.get("raw_header_lines", [])
    has_dark = bool(dark_by_label and any(v is not None for v in dark_by_label.values()))

    with h5py.File(output_path, "w") as f:
        # ── Primary datasets ───────────────────────────────────────
        d_wl = f.create_dataset("Energy", data=ev_out, compression="gzip")
        d_wl.attrs["units"] = "eV"

        d_diff = f.create_dataset("SpectraDiff", data=counts_diff, compression="gzip")
        d_diff.attrs["units"]           = f"Counts/{int_time_str}s"
        d_diff.attrs["axes"]            = "Energy : Power_uncalibrated"
        d_diff.attrs["dark_subtracted"] = "Yes" if has_dark else "No"

        d_raw = f.create_dataset("Spectra_raw", data=counts_raw, compression="gzip")
        d_raw.attrs["units"] = f"Counts/{int_time_str}s"
        d_raw.attrs["axes"]  = "Energy : Power_uncalibrated"

        d_p = f.create_dataset("Power_uncalibrated", data=powers_W * 1e3)
        d_p.attrs["units"] = "mW"

        if hwp_vals:
            try:
                hwp_arr = np.array(
                    [float(v) for v in hwp_vals[:len(powers_W)]], dtype=float
                )
                d_h = f.create_dataset("hwp_positions", data=hwp_arr)
                d_h.attrs["label"] = hwp_label
                d_h.attrs["units"] = "degrees"
            except ValueError:
                pass

        # ── Dark spectrum (non-stitched only) ─────────────────────
        if not stitched and has_dark:
            dark_arr = next(iter(dark_by_label.values()))
            d_dk = f.create_dataset("darkspec", data=dark_arr, compression="gzip")
            d_dk.attrs["units"] = f"Counts/{int_time_str}s"

        # ── Root attributes ────────────────────────────────────────
        f.attrs["format_version"] = 1
        f.attrs["stitched"]       = "Yes" if stitched else "No"
        f.attrs["date_str"]       = header_meta.get("date_str", "")
        f.attrs["int_time_str"]   = int_time_str
        f.attrs["center_nm"]      = center_nm
        f.attrs["center_eV"]      = center_eV
        f.attrs["dispersion_nm"]  = disp_nm
        f.attrs["dispersion_eV"]  = disp_eV
        if spot_diameter_um is not None:
            f.attrs["spot_diameter_um"] = float(spot_diameter_um)
        if rep_rate_mhz is not None:
            f.attrs["rep_rate_mhz"] = float(rep_rate_mhz)
        for i, line in enumerate(raw[:9]):
            f.attrs[f"header_line_{i}"] = line.rstrip("\r\n")

        # ── Source spectra (stitched files only) ───────────────────
        if stitched and source_datasets:
            grp = f.create_group("source_spectra")
            for d in source_datasets:
                lbl = d["label"].replace("/", "_")
                sg = grp.create_group(lbl)
                sg_wl = sg.create_dataset(
                    "wavelength_nm", data=d["wl"], compression="gzip"
                )
                sg_wl.attrs["units"] = "nm"
                sg_c = sg.create_dataset(
                    "counts", data=d["counts"], compression="gzip"
                )
                sg_c.attrs["units"] = f"Counts/{int_time_str}s"
                if dark_by_label:
                    dark_arr = dark_by_label.get(d["label"])
                    if dark_arr is not None:
                        sg_dk = sg.create_dataset(
                            "darkspec", data=dark_arr, compression="gzip"
                        )
                        sg_dk.attrs["units"] = f"Counts/{int_time_str}s"

        # ── Power calibration ─────────────────────────────────────
        # power used for derived quantities; updated to calibrated value if available
        power_for_derived = powers_W

        if power_cal is not None:
            grp_c = f.create_group("power_calibration")
            for role in ("atBS", "atSample"):
                entry = power_cal.get(role)
                if entry is not None:
                    hwp_arr, pow_arr = entry
                    dh = grp_c.create_dataset(f"{role}_hwp_positions", data=hwp_arr)
                    dh.attrs["units"] = "degrees"
                    dp = grp_c.create_dataset(f"{role}_power_W", data=pow_arr)
                    dp.attrs["units"] = "W"

            # Compute Power when both sides are present.
            # Method mirrors MATLAB: fit  P_sample = a * P_BS  (linear through
            # origin, OLS) using the overlapping HWP positions, then apply
            # P_calibrated = a * Power_uncalibrated.
            cal_bs  = power_cal.get("atBS")
            cal_s   = power_cal.get("atSample")
            if cal_bs is not None and cal_s is not None:
                hwp_bs,  pow_bs  = cal_bs
                hwp_s,   pow_s   = cal_s
                hwp_bs_r = np.round(hwp_bs).astype(int)
                hwp_s_r  = np.round(hwp_s).astype(int)
                common   = np.intersect1d(hwp_bs_r, hwp_s_r)
                if len(common) >= 2:
                    idx_bs = np.array([np.where(hwp_bs_r == h)[0][0] for h in common])
                    idx_s  = np.array([np.where(hwp_s_r  == h)[0][0] for h in common])
                    x = pow_bs[idx_bs]
                    y = pow_s [idx_s ]
                    a_transmission  = float(np.dot(x, y) / np.dot(x, x))
                    power_for_derived = a_transmission * powers_W
                    d_pc = f.create_dataset("Power", data=power_for_derived * 1e3)
                    d_pc.attrs["units"]         = "mW"
                    d_pc.attrs["transmission"]  = a_transmission
                    d_pc.attrs["n_cal_points"]  = len(common)
                    grp_c.attrs["transmission"] = a_transmission

        # ── Derived spatial/temporal quantities ───────────────────
        if spot_diameter_um is not None:
            dspot_cm = float(spot_diameter_um) * 1e-4          # µm → cm
            power_density = 4.0 * power_for_derived / np.pi / dspot_cm ** 2
            d_pd = f.create_dataset("Power_density", data=power_density * 1e-3)
            d_pd.attrs["units"] = "kW/cm^2"

            if rep_rate_mhz is not None:
                f_hz    = float(rep_rate_mhz) * 1e6            # MHz → Hz
                fluence = (4.0 * power_for_derived / f_hz
                           / np.pi / dspot_cm ** 2 * 1e3)      # → mJ/cm²
                d_fl = f.create_dataset("Pump_fluence", data=fluence)
                d_fl.attrs["units"] = "mJ/cm^2"


def _parse_power_calibration(filepath):
    """Parse an atBS or atSample power calibration .origin file.

    The file stores rows of  <power_W>  <hwp_pos>  <power_W>  (tab-separated).
    Returns (hwp_positions, powers_W) sorted by HWP position.
    """
    hwp, pows = [], []
    with open(filepath, encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                pw = float(parts[0])
                hp = float(parts[1])
                hwp.append(hp)
                pows.append(pw)
            except ValueError:
                continue
    if not hwp:
        raise ValueError(f"No numeric data found in {filepath}")
    hwp  = np.array(hwp,  dtype=float)
    pows = np.array(pows, dtype=float)
    order = np.argsort(hwp)
    return hwp[order], pows[order]
