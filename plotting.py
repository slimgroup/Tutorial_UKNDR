from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from shapely.geometry import Polygon


def plot_boundary_with_wells(ds: xr.Dataset, x: np.ndarray, y: np.ndarray, names: np.ndarray):
    """Create an axes with survey boundary and well locations.

    Returns an axes so callers can further draw or extract the boundary polygon.
    """
    ax = ds.segysak.plot_bounds()
    ax.plot(x, y, "ro", label="Wells")
    for xi, yi, label in zip(x, y, names):
        ax.text(
            xi,
            yi,
            label,
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )
    return ax


def boundary_polygon_from_axes(ax) -> Polygon:
    """Build a polygon from the first line in the given axes (survey boundary)."""
    boundary_xy = np.c_[ax.lines[0].get_xdata(), ax.lines[0].get_ydata()]
    return Polygon(boundary_xy)


def plot_splines(ax, sequences: List[Tuple[np.ndarray, np.ndarray]]):
    """Overlay a list of spline polylines onto the provided axes."""
    cmap = plt.get_cmap("hsv")
    total = max(1, len(sequences))
    for idx, (xs, ys) in enumerate(sequences, start=1):
        ax.plot(xs, ys, "-", color=cmap(idx / total), alpha=0.85, linewidth=1.2)
    return ax


def save_boundary_with_wells(ds: xr.Dataset, x: np.ndarray, y: np.ndarray, names: np.ndarray, out_path: str, show: bool = False) -> None:
    """Render boundary+wells to file without disrupting the calling flow."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = ds.segysak.plot_bounds(ax=ax)
    ax.plot(x, y, "ro", label="Wells")
    # Ensure out-of-bound wells are still visible within the figure view
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        wxmin, wxmax = float(np.nanmin(x)), float(np.nanmax(x))
        wymin, wymax = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 200.0
        ax.set_xlim(min(xmin, wxmin) - pad, max(xmax, wxmax) + pad)
        ax.set_ylim(min(ymin, wymin) - pad, max(ymax, wymax) + pad)
    except Exception:
        pass
    for xi, yi, label in zip(x, y, names):
        ax.text(
            xi,
            yi,
            label,
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )
    ax.legend(["Wells"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_splines_overview(ds: xr.Dataset, x: np.ndarray, y: np.ndarray, sequences: List[Tuple[np.ndarray, np.ndarray]], out_path: str, show: bool = False) -> None:
    """Render boundary+wells and all spline paths to a file."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = ds.segysak.plot_bounds(ax=ax)
    ax.plot(x, y, "ro", label="Wells")
    # Ensure out-of-bound wells are still visible within the figure view
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        wxmin, wxmax = float(np.nanmin(x)), float(np.nanmax(x))
        wymin, wymax = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 200.0
        ax.set_xlim(min(xmin, wxmin) - pad, max(xmax, wxmax) + pad)
        ax.set_ylim(min(ymin, wymin) - pad, max(ymax, wymax) + pad)
    except Exception:
        pass
    plot_splines(ax, sequences)
    ax.legend(["Wells"], ncol=2, fontsize=8)
    ax.set_title("Splines Through Wells (Extended & Clipped Inside Boundary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_line_overview(ds: xr.Dataset, x: np.ndarray, y: np.ndarray, line_coords: np.ndarray, line_016, line_name: str, out_path: str | None = None, show: bool = False) -> None:
    """Render an overview map for a specific spline line with sampled trace locations."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = ds.segysak.plot_bounds(ax=ax)
    ax.plot(x, y, "ro", label="Wells")
    ax.plot(line_coords[:, 0], line_coords[:, 1], "b-", linewidth=2, label=line_name)
    trace_x, trace_y = line_016["cdp_x"].values, line_016["cdp_y"].values
    ax.plot(trace_x, trace_y, "k.", markersize=2, label="Trace locations")
    ax.legend()
    ax.set_title(f"{line_name} with Extracted Traces Along Path")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


