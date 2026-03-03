import pathlib
import os
from time import time
from typing import Dict, List, Tuple

import lasio
import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from scipy.signal import convolve

#from . import dataset as ds_writer
import io
#from . import mapping
import plotting
import seismic
import splines
#from . import synthetic
#from . import velocity
from .config import Config, load_config

def _preload_las_files(las_paths):
    cache = {}
    for label, path in las_paths.items():
        if not path or not os.path.exists(path):
            cache[label] = None
            continue
        try:
            cache[label] = lasio.read(path)
        except Exception:
            cache[label] = None

    return cache


def _find_intersecting_wells(line_coords, well_df, tol=0.0):
    spline_line = LineString(line_coords)
    hits = []

    for wname, wx, wy in zip(well_df["Nickname"], well_df["X"], well_df["Y"]):
        pt = Point(wx, wy)
        if spline_line.intersects(pt) or spline_line.distance(pt) <= tol:
            hits.append((wname, float(wx), float(wy)))

    return hits



def _build_synthetic_for_wells(intersecting_wells, las_cache, cfg):
    synthetic_data = {}

    depth_keys = ["DEPTH", "DEPT", "BOREHOLE-DEPTH"]
    dt_keys = ["DT", "AC", "DT4P"]

    # Precompute wavelet once (same for all wells)
    w, _ = synthetic.ricker_wavelet(
        duration=cfg.process.ricker_duration_s,
        dt=cfg.process.ricker_dt_s,
        f0=cfg.process.ricker_f0_hz,
    )

    for wname, wx, wy in intersecting_wells:
        label = wname[-1].upper()
        print(f"label: {label}")
        las = las_cache.get(label)
        print(f"las: {las}")
        if las is None:
            print(f"LAS missing for {wname}")
            continue

        # ---- Depth curve & unit ----
        depth_curve = None
        depth_unit = None
        for k in depth_keys:
            if k in las.keys():
                depth_curve = np.asarray(las[k])
                if k in las.curves and las.curves[k].unit:
                    depth_unit = las.curves[k].unit.strip().lower()
                break

        if depth_curve is None:
            print("No depth curve found")
            continue

        if depth_unit in ["ft", "feet", "foot"]:
            depth_m = depth_curve * 0.3048
        elif depth_unit in ["m", "M", "meter", "metre", "Meters"]:
            depth_m = depth_curve
        else:
            depth_m = depth_curve * 0.3048 if depth_curve.max() > 3500 else depth_curve

        # ---- DT curve & unit ----
        dt_curve = None
        dt_unit = "unknown"
        for key in las.keys():
            key_stripped = key.strip().upper()
            if key_stripped in dt_keys:
                dt_curve = np.asarray(las[key])
                if key in las.curves and las.curves[key].unit:
                    dt_unit = las.curves[key].unit.strip().lower()
                break

        if dt_curve is None:
            print("No DT/AC found")
            continue

        # Normalize dt to seconds per meter
        if dt_unit in ["us/ft", "µs/ft", "microsec/ft", "us/f", "US/F"]:
            dt_s_per_m = dt_curve * 1e-6 / 0.3048
        elif dt_unit in ["us/m", "µs/m", "microsec/m"]:
            dt_s_per_m = dt_curve * 1e-6
        else:
            dt_s_per_m = dt_curve * (1e-6 / 0.3048 if np.nanmean(dt_curve) > 200 else 1e-6)

        # ---- Velocity cleanup ----
        velocity_raw = np.where(dt_s_per_m > 0, 1.0 / dt_s_per_m, np.nan)
        mask_valid_velocity = np.isfinite(velocity_raw) & np.isfinite(depth_m)
        depth_m_clean = depth_m[mask_valid_velocity]
        velocity_full = velocity_raw[mask_valid_velocity]

        if len(velocity_full) < 5:
            print("Not enough valid velocity points")
        print(f"velocity_full: {velocity_full}")
        print(f"velocity_full length: {len(velocity_full)}")

        synthetic_data[wname] = {
            "depth_m": depth_m_clean,
            "velocity_m_s_full": velocity_full,
            "coords": (wx, wy),
        }

    return synthetic_data



def _build_h5_attrs(depth_resampled):
    """Metadata for HDF5 root attributes."""
    return {
        "description": "Aligned seismic (256×512), velocity (256×512), and per-well downsampled velocity logs with unique pixel mappings and reflectivity-valid region.",
        "created_by": "Ipsita Bhar",
        "depth_pixel_range": "0–255",
        "trace_pixel_range": "0–511",
        "depth_range_m": f"{depth_resampled.min():.1f}–{depth_resampled.max():.1f} m",
    }


def _build_h5_globals(depth_resampled, out_nx, distance_resampled, seis_resampled, x_cdp, y_cdp):
    """Global 2D arrays written once per line."""
    return {
            "depth_resampled": depth_resampled,
            "trace_index_resampled": np.arange(out_nx),
            "distance_resampled": distance_resampled,
            "seismic_section_full": seis_resampled,
            "cdp_x": x_cdp,
            "cdp_y": y_cdp,
        }


def _build_h5_per_well(
    synthetic_data,
    nearest_ilines,
    line_016,
    out_nx,
    out_nz,
    depth_resampled,
    dataset_name,
):
    per_well = {}

    cdp_x = line_016["cdp_x"].values
    cdp_y = line_016["cdp_y"].values
    seismic_section = line_016["data"].values.T

    zmin = float(depth_resampled.min())
    zmax = float(depth_resampled.max())
    z_axis = np.arange(out_nz)

    for wname, wdata in synthetic_data.items():

        wx, wy = wdata["coords"]

        n_cols_resampled = out_nx
        n_cols_orig_clean = len(cdp_x)
        trace_idx = mapping.nearest_trace_index(wx, wy, cdp_x, cdp_y)
        trace_idx_resampled =  int(np.clip(int(round(trace_idx * (n_cols_resampled / max(1, n_cols_orig_clean)))), 0, n_cols_resampled - 1))


        vel_full = wdata["velocity_m_s_full"]
        vel_full = np.where(np.isfinite(vel_full), vel_full, np.nanmean(vel_full)) 

        # Vertical window along seismic depth axis
        w_depth = wdata["depth_m"]
        mask_win = (w_depth >= zmin) & (w_depth <= zmax)

        if np.any(mask_win):
            w_depth_win = w_depth[mask_win]
            pix_bounds = np.interp(
                [float(np.nanmin(w_depth_win)), float(np.nanmax(w_depth_win))],
                depth_resampled,
                z_axis,
                left=np.nan,
                right=np.nan,
            )
            if np.isfinite(pix_bounds[0]):
                depth_pixel_start = int(np.clip(int(np.ceil(pix_bounds[0])), 0, out_nz - 1))
            else:
                depth_pixel_start = 0

            if np.isfinite(pix_bounds[1]):
                depth_pixel_end = int(np.clip(int(np.floor(pix_bounds[1])), 0, out_nz - 1))
            else:
                depth_pixel_end = 0
        else:
            w_depth_win = np.array([], dtype=float)
            depth_pixel_start, depth_pixel_end = 0, 0

        covered_len_px = max(1, depth_pixel_end - depth_pixel_start + 1)

        # Resample well velocity to EXACTLY match covered_len_px pixels
        n_depth = len(w_depth)
        if n_depth >= 2:
            depth_lo = float(np.nanmin(w_depth_win)) if np.any(mask_win) else float(np.nanmin(w_depth))
            depth_hi = float(np.nanmax(w_depth_win)) if np.any(mask_win) else float(np.nanmax(w_depth))
            order = np.argsort(w_depth)
            w_depth_sorted = w_depth[order]
            vel_sorted = vel_full[order]
            depth_full_ds = np.linspace(depth_lo, depth_hi, covered_len_px)
            vel_full_ds = np.interp(depth_full_ds, w_depth_sorted, vel_sorted)
        elif n_depth == 1:
            depth_full_ds = np.linspace(w_depth.min(), w_depth.max(), covered_len_px)
            vel_full_ds = np.full(covered_len_px, float(vel_full[0]))
        else:
            depth_full_ds = np.linspace(zmin, zmax, covered_len_px)
            vel_full_ds = np.full(covered_len_px, np.nan)

        slow_full_ds = np.where(vel_full_ds > 0, 1.0 / vel_full_ds, np.nan)

        # Pixel indices align 1:1 with resampled well samples
        well_pixel_idx_ds = np.arange(depth_pixel_start, depth_pixel_end + 1, dtype=np.int32)
        per_well[wname] = {
            "attrs": {
                "x_coord": wx,
                "y_coord": wy,
                "nearest_trace_index_resampled": trace_idx_resampled,
                "depth_pixel_start": depth_pixel_start,
                "depth_pixel_end": depth_pixel_end,
                "nearest_inline_value": float(nearest_ilines[wname]),
                "dataset_name": dataset_name,
            },
            "datasets": {
                "seismic_section": seismic_section,
                "depth_full": wdata["depth_m"],
                "depth_full_downsampled": depth_full_ds,
                "velocity_full_original": vel_full,
                "velocity_full_downsampled": vel_full_ds,
                "slowness_full_downsampled": slow_full_ds,
                "well_pixel_idx_ds": well_pixel_idx_ds,
            },
        }

    return per_well


def _save_h5(out_path, attrs, globals_2d, per_well, enable):
    """Write HDF5 if enabled, otherwise no-op while reporting path."""
    if enable:
        ds_writer.save_h5_bundle(out_path, attrs, globals_2d, per_well)
    print(f"Saved H5 → {out_path}")

"""Run the end-to-end 3-well spline pipeline.

Steps:
    1) Load config, seismic cube, and wells
    2) Save boundary+wells plot; build clipped splines from 3-well combinations
    3) Save splines overview; select N splines to process
    4) For each spline: extract seismic, normalize, resample, and plot overview
    5) Find wells on the spline; generate synthetic from LAS; map to inlines
    6) Extract velocity sections; assemble arrays and (optionally) write HDF5
"""
def run_3wells_pipeline(config_path):
    # Resolve config (path default from repository config folder)
    if config_path is None:
        config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "ukndr.yaml"
    cfg = load_config(str(config_path))

    dataset_name = cfg.process.dataset_name

    # 1) Load seismic and well data once (reused later)
    depth_seismic = io.load_segy_dataset(pathlib.Path(cfg.paths.depth_segy_path))
    well_df, x, y, names = io.load_well_locations(pathlib.Path(cfg.paths.wells_csv_path))
    # Resolve figures directory from config and ensure it exists
    figures_dir = pathlib.Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # after loading depth_seismic
    inline_spacing_m = float(depth_seismic.attrs.get("inline_spacing_m", float("nan")))
    print(f"Inline spacing (m): {inline_spacing_m}")

    # Preload LAS file handles once; synthetic generation later will reuse this
    las_paths = (cfg.paths.las_paths or {})
    las_cache = _preload_las_files(las_paths)
    print(f"las_cache: {las_cache}")

    # 2) Boundary+wells plot (saved; non-blocking)
    plotting.save_boundary_with_wells(
        depth_seismic, x.values, y.values, names.values, out_path=str(figures_dir / f"{dataset_name}_well_boundary_plot.png"), show=False
    )

    # Construct boundary polygon once. This polygon is used to clip splines.
    ax_boundary = depth_seismic.segysak.plot_bounds()
    poly = plotting.boundary_polygon_from_axes(ax_boundary)
    plt.close(ax_boundary.figure)

    # Build requested well-order splines (3..9) per config; fallback to 3-well default
    lines_all = []
    spline_sequences = []

    orders_request = cfg.process.spline_orders or {}
    if not orders_request:
        # Fallback to original triplets and num_splines_to_process
        triplets = splines.build_triplets(len(x))
        take = min(len(triplets), cfg.process.num_splines_to_process)
        for item_idx, comb in enumerate(triplets[:take], start=1):
            pts = np.array([[x.values[idx], y.values[idx]] for idx in comb])
            # Deterministic per-line RNG: base seed + item index
            base_seed = int(cfg.process.spline_extension_seed) if cfg.process.spline_extension_seed is not None else 0
            xs, ys = splines.fit_parametric_spline_through_points(
                pts,
                n=200,
                method="bspline",
                k=None,
                extension_ratio=0.25,
                rng_seed=base_seed + item_idx,
                max_extension_m=float(cfg.process.spline_extension_max_m or 0.0),
                poly=poly,
            )
            coords = splines.force_path_through_wells(xs, ys, pts, keep_tails=True)
            xs_c, ys_c = splines.clip_spline_to_boundary(coords[:, 0], coords[:, 1], poly, points=pts)
            coords = np.column_stack([xs_c, ys_c])

            # xs, ys = splines.clip_spline_to_boundary(xs, ys, poly)
            # coords = splines.force_path_through_wells(xs, ys, pts)
            # Diagnostics: distances from design wells to path
            dists = splines.point_to_path_distances(coords[:, 0], coords[:, 1], pts)
            if np.nanmax(dists) > 100.0:
                print(f"[WARN] Max well-to-path distance {np.nanmax(dists):.1f} m for {name}")
            name = "Spline_" + "-".join(str(idx) for idx in comb)
            lines_all.append({"Line_Name": name, "Wells": comb, "CoordArray": coords})
            spline_sequences.append((coords[:, 0], coords[:, 1]))
    else:
        # Build per requested order with validation
        for order, count in sorted(orders_request.items()):
            if order < 3 or order > 11:
                raise ValueError(f"spline_orders key {order} is out of supported range [3, 11]")
            combs = splines.build_combinations(len(x), order)
            if count > len(combs):
                raise ValueError(f"Requested {count} splines for order {order}, but only {len(combs)} combinations available")
            for item_idx, comb in enumerate(combs[:count], start=1):
                pts = np.array([[x.values[idx], y.values[idx]] for idx in comb])
                base_seed = int(cfg.process.spline_extension_seed) if cfg.process.spline_extension_seed is not None else 0
                xs, ys = splines.fit_parametric_spline_through_points(
                    pts,
                    n=200,
                    method="bspline",
                    k=None,
                    extension_ratio=0.25,
                    rng_seed=base_seed + item_idx,
                    max_extension_m=float(cfg.process.spline_extension_max_m or 0.0),
                    poly=poly,
                )
                coords = splines.force_path_through_wells(xs, ys, pts, keep_tails=True)
                # xs_c, ys_c = splines.clip_spline_to_boundary(coords[:, 0], coords[:, 1], poly, points=pts)
                # coords = np.column_stack([xs_c, ys_c])

                # This was causing troubles with the splines that have loop
                # xs, ys = splines.clip_spline_to_boundary(xs, ys, poly, points=pts)
                # coords = splines.force_path_through_wells(xs, ys, pts)
                # Diagnostics: distances from design wells to path
                dists = splines.point_to_path_distances(coords[:, 0], coords[:, 1], pts)
                if np.nanmax(dists) > 100.0:
                    print(f"[WARN] Max well-to-path distance {np.nanmax(dists):.1f} m for {name}")
                name = "Spline_" + "-".join(str(idx) for idx in comb)
                lines_all.append({"Line_Name": name, "Wells": comb, "CoordArray": coords})
                spline_sequences.append((coords[:, 0], coords[:, 1]))

    # 3) Save overview figure of all generated splines
    plotting.save_splines_overview(
        depth_seismic,
        x.values,
        y.values,
        spline_sequences,
        out_path=str(figures_dir / f"{dataset_name}_all_splines_overview.png"),
        show=False,
    )


    lines_df = pd.DataFrame(lines_all)

    # Number of splines to process for dataset creation
    if orders_request:
        selected_line_names = lines_df["Line_Name"].tolist()
    else:
        n = cfg.process.num_splines_to_process
        selected_line_names = lines_df["Line_Name"].tolist()[:n]
    print(f"Preparing to process {len(selected_line_names)} splines...")

    # (helper moved to splines.force_path_through_wells to keep pipeline minimal)

    for idx, line_name in enumerate(selected_line_names, 1):
        print("\n========== Processing", line_name, f"({idx}/{len(selected_line_names)}) ==========")

        line_coords = lines_df.loc[
            lines_df["Line_Name"] == line_name, "CoordArray"
        ].values[0]

        # 4) Extract seismic along spline and create diagnostic overview
        tic = time()

        # Guard against NaNs/degenerate paths before calling interp_line
        if not np.isfinite(line_coords).all():
            mask_finite = np.isfinite(line_coords).all(axis=1)
            line_coords = line_coords[mask_finite]

        # Drop consecutive duplicate points
        if len(line_coords) >= 2:
            diffs = np.diff(line_coords, axis=0)
            keep = np.insert(np.any(diffs != 0.0, axis=1), 0, True)
            line_coords = line_coords[keep]

        # Fallback: if still invalid/too short, use straight polyline through design wells
        if len(line_coords) < 2:
            comb_used_fallback = lines_df.loc[
                lines_df["Line_Name"] == line_name, "Wells"
            ].values[0]
            pts_fallback = np.array([[x.values[i], y.values[i]] for i in comb_used_fallback])
            line_coords = pts_fallback
        else:
            seglens = np.sqrt(np.sum(np.diff(line_coords, axis=0) ** 2, axis=1))
            total_len = float(np.nansum(seglens)) if np.isfinite(seglens).any() else 0.0
            if not np.isfinite(total_len) or total_len <= 0.0:
                comb_used_fallback = lines_df.loc[
                    lines_df["Line_Name"] == line_name, "Wells"
                ].values[0]
                pts_fallback = np.array([[x.values[i], y.values[i]] for i in comb_used_fallback])
                line_coords = pts_fallback

        line_016 = seismic.interp_line(depth_seismic, line_coords, bin_spacing_hint=10)
        print(f"interp_line took {time() - tic:.2f} s")

        plotting.save_line_overview(
            depth_seismic,
            x.values,
            y.values,
            line_coords,
            line_016,
            line_name,
            out_path=str(figures_dir / f"{dataset_name}_spline_well_overview_{line_name}.png"),
            show=False,
        )


        # Seismic section arrays
        seis = line_016["data"].values.T
        depth = np.asarray(line_016["samples"].values)
        inlines = line_016["iline"].values

        # Crop to configured maximum depth and normalize symmetrically
        seis_cropped, depth_cropped = seismic.crop_depth(seis, depth, cfg.process.max_depth_m)
        seis_scaled, norm = seismic.normalize_symmetric(seis_cropped)

        # Get sample shape and inlines
        out_nz = cfg.process.resample_nz
        out_nx = cfg.process.resample_nx

        # Drop any columns with NaN
        nan_cols = np.where(np.isnan(seis_scaled).any(axis=0))
        seis_scaled = np.delete(seis_scaled, nan_cols, axis=1)

        # Subsample the cleaned array
        #seis_resampled_y = resample(seis_scaled_clean, out_nz, axis=0)
        #depth_resampled = np.linspace(depth_cropped.min(), depth_cropped.max(), out_nz)
        #seis_resampled = resample(seis_resampled_y, out_nx, axis=1)

        # Replace the existing Z resample with:
        seis_resampled, depth_resampled = seismic.resample_section_2d_fft_lowpass(seis_scaled, depth_cropped, edge_taper_frac=0.25, out_nz=cfg.process.resample_nz, out_nx=cfg.process.resample_nx)
        #seis_resampled, depth_resampled, _ = seismic.resample_section(seis_scaled, depth_cropped, out_nz=cfg.process.resample_nz, out_nx=cfg.process.resample_nx)
        # seis_resampled, depth_resampled = seismic.resample_section_freq_taper_z(
        #     seis_scaled,
        #     depth_cropped,
        #     out_nz=cfg.process.resample_nz,
        #     out_nx=cfg.process.resample_nx,
        #     taper_start_frac=0.025,   # begin taper at 60% of Nyquist
        #     taper_end_frac=0.40,     # fully suppressed by 90% of Nyquist
        #     taper_power=5.0         # steeper attenuation (try 3–6)
        # )
        inlines_resampled = np.linspace(inlines.min(), inlines.max(), out_nx)

        # # Quick spectral check: mid column (along depth) and mid row (along traces)
        # def _amp_spectrum(sig: np.ndarray):
        #     s = np.nan_to_num(sig - np.nanmean(sig))
        #     spec = np.abs(np.fft.rfft(s))
        #     freq = np.fft.rfftfreq(len(s), d=1.0)  # normalized cycles/sample
        #     spec = spec / (np.max(spec) + 1e-12)
        #     return freq, spec

        # try:
        #     # Mid column spectra (Z direction)
        #     c0 = max(0, min(seis_scaled.shape[1] // 2, seis_scaled.shape[1] - 1))
        #     c1 = max(0, min(seis_resampled.shape[1] // 2, seis_resampled.shape[1] - 1))
        #     fz0, az0 = _amp_spectrum(seis_scaled[:, c0])
        #     fz1, az1 = _amp_spectrum(seis_resampled[:, c1])
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        #     ax[0].plot(fz0, az0, color="tab:blue")
        #     ax[0].set_title("orig (mid col)")
        #     ax[0].set_xlabel("Norm. freq")
        #     ax[0].set_ylabel("Amp (norm.)")
        #     ax[1].plot(fz1, az1, color="tab:orange")
        #     ax[1].set_title("down (mid col)")
        #     ax[1].set_xlabel("Norm. freq")
        #     fig.suptitle(f"{dataset_name} {line_name} – Column spectrum", y=1.02)
        #     fig.tight_layout()
        #     fig.savefig(str(figures_dir / f"{dataset_name}_{line_name}_spectrum_col_mid.png"), dpi=150)
        #     plt.close(fig)

        #     # Mid row spectra (X direction)
        #     r0 = max(0, min(seis_scaled.shape[0] // 2, seis_scaled.shape[0] - 1))
        #     r1 = max(0, min(seis_resampled.shape[0] // 2, seis_resampled.shape[0] - 1))
        #     fx0, ax0 = _amp_spectrum(seis_scaled[r0, :])
        #     fx1, ax1 = _amp_spectrum(seis_resampled[r1, :])
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        #     ax[0].plot(fx0, ax0, color="tab:blue")
        #     ax[0].set_title("orig (mid row)")
        #     ax[0].set_xlabel("Norm. freq")
        #     ax[0].set_ylabel("Amp (norm.)")
        #     ax[1].plot(fx1, ax1, color="tab:orange")
        #     ax[1].set_title("down (mid row)")
        #     ax[1].set_xlabel("Norm. freq")
        #     fig.suptitle(f"{dataset_name} {line_name} – Row spectrum", y=1.02)
        #     fig.tight_layout()
        #     fig.savefig(str(figures_dir / f"{dataset_name}_{line_name}_spectrum_row_mid.png"), dpi=150)
        #     plt.close(fig)
        # except Exception as _e:
        #     print(f"[warn] spectral check failed: {_e}")

        # Diagnostic plot: cropped + normalized section
        plt.figure(figsize=(12, 6))
        plt.imshow(
            seis_scaled,
            cmap="gray",
            aspect="auto",
            extent=[
                0,
                (inlines.max() - inlines.min()) * inline_spacing_m,
                depth_cropped.max(),
                depth_cropped.min(),
            ],
            norm=norm,
            origin="upper",
        )
        plt.title("Seismic Section (0–3200 m)", fontsize=13)
        plt.xlabel("Inline (m)")
        plt.ylabel("Depth (m)")
        ax = plt.gca()
        yticks = ax.get_yticks()
        ax.set_yticklabels(yticks[::-1])
        plt.colorbar(label="Normalized Amplitude")
        plt.tight_layout()
        plt.savefig(str(figures_dir / f"{dataset_name}_seismic_{line_name}.jpg"))
        plt.close()

        # Diagnostic plot: resampled section
        plt.figure(figsize=(12, 6))
        plt.imshow(
            seis_resampled,
            cmap="gray",
            aspect="auto",
            extent=[
                0,
                (inlines_resampled.max() - inlines_resampled.min()) * inline_spacing_m,
                depth_resampled.min(),
                depth_resampled.max(),
            ],
            norm=norm,
            origin="upper",
        )
        plt.title(f"Seismic Section (0–3200 m) {line_name} 256x512", fontsize=13)
        plt.xlabel("Inline (m)")
        plt.ylabel("Depth (m)")
        ax = plt.gca()
        yticks = ax.get_yticks()
        ax.set_yticklabels(yticks[::-1])
        plt.colorbar(label="Normalized Amplitude")
        plt.tight_layout()
        plt.savefig(str(figures_dir / f"{dataset_name}_seismic_{line_name}_resampled.jpg"))
        plt.close()

        # 5) Intersect wells and gather candidates
        # Always use exactly the wells used to build this spline (design wells),
        # preventing accidental inclusion of nearby wells.
        comb_used = lines_df.loc[lines_df["Line_Name"] == line_name, "Wells"].values[0]
        design_wells = [(names.values[i], float(x.values[i]), float(y.values[i])) for i in comb_used]
        intersecting_wells = design_wells

        if intersecting_wells:
            print(f"\nWells intersecting or near {line_name}:")
            for wname, wx, wy in intersecting_wells:
                print(f"  {wname:<6} (X={wx:.2f}, Y={wy:.2f})")
        else:
            print(f"No wells intersect or lie near {line_name}.")

        # Build synthetic data per intersecting well using preloaded LAS files
        synthetic_data = _build_synthetic_for_wells(intersecting_wells, las_cache, cfg)

        # 6) Map wells to inline using KDTree
        x_cdp = line_016["cdp_x"].values
        y_cdp = line_016["cdp_y"].values

        # Remove any rows with NaNs jointly to preserve alignment with iline
        iline_vals = line_016["iline"].values
        finite_mask = np.isfinite(x_cdp) & np.isfinite(y_cdp) & np.isfinite(iline_vals)
        x_cdp = x_cdp[finite_mask]
        y_cdp = y_cdp[finite_mask]
        iline_vals = iline_vals[finite_mask]

        # keep first index per (rounded) inline value; ensures increasing inline order
        # u_il, first_idx = np.unique(np.rint(iline).astype(int), return_index=True)
        # x_il = x[first_idx]; y_il = y[first_idx]

        # spacing_inline_m = np.hypot(np.diff(x_il), np.diff(y_il))  # length = n_unique_inlines-1
        # print(float(np.mean(spacing_inline_m)), float(np.std(spacing_inline_m)))

        plotting.save_line_overview(
            depth_seismic, x_cdp, y_cdp, line_coords, line_016, line_name, out_path=str(figures_dir / f"test_splines_well_overview_{line_name}.png"), show=True
        )
        # print(x_cdp, y_cdp, iline_vals)
        cdp_tree = mapping.build_cdp_kdtree(x_cdp, y_cdp)
        # Map all design wells (not only those with LAS) to nearest inline
        well_to_inline = mapping.map_wells_to_nearest_inline(
            cdp_tree,
            iline_vals,
            [(w, wx, wy) for (w, wx, wy) in intersecting_wells],
        )
        for w, info in well_to_inline.items():
            print(f"{w:<6} → Inline {info['inline']} (distance {info['dist_m']:.1f} m)")

        nearest_ilines = {w: info["inline"] for w, info in well_to_inline.items()}

        # Distance axis along line (meters)
        dx = np.diff(line_016["cdp_x"].values)
        dy = np.diff(line_016["cdp_y"].values)
        distance_along = np.insert(np.cumsum(np.sqrt(dx ** 2 + dy ** 2)), 0, 0.0)
        distance_resampled = np.linspace(distance_along.min(), distance_along.max(), out_nx)

        # Save H5 per line (disabled by default; toggle enable=True to write)
        h5_path = pathlib.Path(cfg.paths.h5_output_dir) / f"{dataset_name}_{line_name}.h5"
        globals_2d = _build_h5_globals(
            depth_resampled,
            out_nx,
            distance_resampled,
            seis_resampled,
            line_016["cdp_x"].values,
            line_016["cdp_y"].values,
        )
        per_well = _build_h5_per_well(synthetic_data, nearest_ilines, line_016, out_nx, out_nz, depth_resampled, dataset_name)
        attrs = _build_h5_attrs(depth_resampled)
        # change enable=True to actually write files
        
        print(f"Writing H5 → {h5_path}")
        # Ensure output directory exists
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        ds_writer.save_h5_bundle(h5_path, attrs, globals_2d, per_well)
        print(f"Saved H5 → {h5_path}")

        # Minimal overlay plot: seismic with well velocities (km/s) using cc.cm['rainbow4']
        try:
            vel_canvas = np.full_like(seis_resampled, np.nan, dtype=float)
            # Use cleaned CDP coordinates (NaNs removed) for robust column mapping
            n_cols_resampled = seis_resampled.shape[1]
            n_cols_orig_clean = len(x_cdp)
            def col_for_xy(wx, wy):
                idx_orig = mapping.nearest_trace_index(wx, wy, x_cdp, y_cdp)
                return int(np.clip(int(round(idx_orig * (n_cols_resampled / max(1, n_cols_orig_clean)))), 0, n_cols_resampled - 1))

            # Paint velocity where available; compute column via cleaned coords
            for wname, wentry in per_well.items():
                print(f"well name: {wname}")
                wx = float(wentry["attrs"].get("x_coord", np.nan))
                wy = float(wentry["attrs"].get("y_coord", np.nan))
                if not np.isfinite(wx) or not np.isfinite(wy):
                    continue
                col = col_for_xy(wx, wy)
                dsets = wentry.get("datasets", {})
                rows = dsets.get("well_pixel_idx_ds")
                vvals_ms = dsets.get("velocity_full_downsampled")
                if col < 0 or rows is None or vvals_ms is None:
                    continue
                rows = np.asarray(rows).astype(int)
                vvals_ms = np.asarray(vvals_ms).astype(float)
                L = min(len(rows), len(vvals_ms))
                if L == 0:
                    continue
                rows = np.clip(rows[:L], 0, seis_resampled.shape[0] - 1)
                vvals_kms = vvals_ms[:L] / 1000.0
                print(f"vvals_kms: {vvals_kms}")
                print(f"col: {col}")
                print(f"seis_resampled.shape: {seis_resampled.shape}")
                for c in (col - 1, col, col + 1):
                    if 0 <= c < seis_resampled.shape[1]:
                        vel_canvas[rows, c] = vvals_kms

            finite_seis = np.isfinite(seis_resampled)
            amax = np.percentile(np.abs(seis_resampled[finite_seis]), 98) if finite_seis.any() else np.nanmax(np.abs(seis_resampled))
            vmin_s, vmax_s = -amax, amax
            finite_vel = np.isfinite(vel_canvas)

            vmin_v, vmax_v = 1.5, 4.6

            plt.figure(figsize=(10, 5))
            plt.imshow(seis_resampled, cmap="gray", aspect="auto", origin="upper", vmin=vmin_s, vmax=vmax_s)
            plt.imshow(np.ma.masked_invalid(vel_canvas), cmap=cc.cm["rainbow4"], aspect="auto", origin="upper", vmin=vmin_v, vmax=vmax_v, alpha=0.85)
            plt.title(f"{line_name} – velocities overlaid (km/s)")
            plt.xlabel("Trace index")
            plt.ylabel("Depth pixel")
            cb = plt.colorbar()
            cb.set_label("Velocity (km/s)")
            out_overlay = figures_dir / f"{dataset_name}_{line_name}_overlay_velocity.png"
            plt.savefig(str(out_overlay), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Overlay plot failed: {e}")


