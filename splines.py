import itertools
from typing import List, Sequence, Tuple, Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from scipy.interpolate import splrep, splev, PchipInterpolator


def build_triplets(num_points: int) -> List[Tuple[int, int, int]]:
    return list(itertools.combinations(range(num_points), 3))

def build_combinations(num_points: int, k: int) -> List[Tuple[int, ...]]:
    """Build all index combinations of size k."""
    return list(itertools.combinations(range(num_points), k))

def fit_parametric_spline(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, n: int = 200, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    t_ctrl = np.array([0.0, 0.5, 1.0])
    tx = splrep(t_ctrl, [P0[0], P1[0], P2[0]], k=k, s=0)
    ty = splrep(t_ctrl, [P0[1], P1[1], P2[1]], k=k, s=0)
    tt = np.linspace(0, 1, n)
    xs = splev(tt, tx)
    ys = splev(tt, ty)
    return np.asarray(xs), np.asarray(ys)


def extend_spline(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, extension_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_start = P1 - P0
    v_end = P2 - P1
    P0_ext = P0 - extension_ratio * v_start
    P2_ext = P2 + extension_ratio * v_end
    return P0_ext, P1, P2_ext


def fit_parametric_spline_through(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    n: int = 200,
    extension_ratio: float = 0.25,
    method: str = "pchip",
    k: int = 2,
    s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a smooth spline through P0, P1, P2 extended slightly beyond endpoints.

    Parameters
    - method: "pchip" (shape-preserving) or "bspline" (degree k, smoothing s)
    - k: spline degree for B-spline (1..5 typical). Ignored for PCHIP.
    - s: smoothing for B-spline. 0 means interpolate exactly; larger s increases smoothness.
    """
    # Extend slightly past the endpoints
    v_start = P1 - P0
    v_end = P2 - P1
    P0_ext = P0 - extension_ratio * v_start
    P2_ext = P2 + extension_ratio * v_end

    # Build node list: include original points so the spline passes through them
    pts = np.vstack([P0_ext, P0, P1, P2, P2_ext])  # shape (5, 2)

    # Chord-length parameterization
    segs = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    t = np.concatenate([[0.0], np.cumsum(segs)])

    tt = np.linspace(t[0], t[-1], n)

    if method.lower() == "bspline":
        # Clamp degree to valid range relative to number of points
        k_eff = int(max(1, min(k, len(pts) - 1)))
        tx = splrep(t, pts[:, 0], k=k_eff, s=s)
        ty = splrep(t, pts[:, 1], k=k_eff, s=s)
        xs = splev(tt, tx)
        ys = splev(tt, ty)
    else:
        # Default: shape-preserving cubic without overshoot
        px = PchipInterpolator(t, pts[:, 0])
        py = PchipInterpolator(t, pts[:, 1])
        xs = px(tt)
        ys = py(tt)
    return np.asarray(xs), np.asarray(ys)

def order_points_spatially(points: np.ndarray, method: str = "pca") -> np.ndarray:
    """Return points ordered along their main axis for stable spline fitting."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts
    if method == "pca":
        c = pts - pts.mean(axis=0, keepdims=True)
        # principal axis (right singular vector)
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        axis = vh[0]
        t = pts @ axis
        idx = np.argsort(t)
        return pts[idx]
    # Fallback: greedy nearest-neighbor chain
    used = np.zeros(len(pts), dtype=bool)
    order = [int(np.argmin(pts[:, 0]))]  # start from leftmost by X
    used[order[0]] = True
    for _ in range(len(pts) - 1):
        last = pts[order[-1]]
        j = int(np.argmin(np.where(used, np.inf, np.sum((pts - last) ** 2, axis=1))))
        order.append(j); used[j] = True
    return pts[order]

def fit_parametric_spline_through_points(
    points: np.ndarray,
    n: int = 200,
    method: str = "pchip",
    k: Optional[int] = None,
    extension_ratio: float = 0.25,
    rng_seed: Optional[int] = None,
    max_extension_m: Optional[float] = None,
    poly: Optional[Polygon] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("points must be (N>=3, 2)")
    # NEW: spatially sort wells for stable curve shape
    points = order_points_spatially(points, method="pca")
    P0, Pn = points[0], points[-1]
    # End-segment directions
    v_start = points[0] - points[1]
    v_end = points[-1] - points[-2]
    # Randomized extension length (meters) if requested; otherwise ratio-based
    if max_extension_m is not None and max_extension_m > 0.0:
        rng = np.random.default_rng(rng_seed)
        d0 = float(rng.uniform(0.0, max_extension_m))
        d1 = float(rng.uniform(0.0, max_extension_m))
        n0 = np.linalg.norm(v_start)
        n1 = np.linalg.norm(v_end)
        u0 = (v_start / n0) if n0 > 0 else np.zeros_like(v_start)
        u1 = (v_end / n1) if n1 > 0 else np.zeros_like(v_end)
        P0_ext = P0 + d0 * u0
        Pn_ext = Pn + d1 * u1
    else:
        P0_ext = P0 + extension_ratio * v_start
        Pn_ext = Pn + extension_ratio * v_end
    # If a boundary polygon is given, prevent extending beyond it per-end
    if poly is not None:
        if not poly.covers(Point(P0_ext[0], P0_ext[1])):
            P0_ext = P0
        if not poly.covers(Point(Pn_ext[0], Pn_ext[1])):
            Pn_ext = Pn
    pts = np.vstack([P0_ext, points, Pn_ext])  # (N+2, 2)

    # Chord-length parameterization
    segs = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    t = np.concatenate([[0.0], np.cumsum(segs)])
    tt = np.linspace(t[0], t[-1], n)

    if method.lower() == "bspline":
        # Desired degree ~ wells+2, but must satisfy k < len(pts) and k <= 5 (Fitpack limit)
        desired_k = (len(points) + 2) if k is None else k
        k_eff = int(max(1, min(5, desired_k, len(pts) - 1)))
        tx = splrep(t, pts[:, 0], k=k_eff, s=0)
        ty = splrep(t, pts[:, 1], k=k_eff, s=0)
        xs = splev(tt, tx)
        ys = splev(tt, ty)
    else:
        px = PchipInterpolator(t, pts[:, 0])
        py = PchipInterpolator(t, pts[:, 1])
        xs = px(tt)
        ys = py(tt)
    return np.asarray(xs), np.asarray(ys)

# add or replace the function signature and body
def clip_spline_to_boundary(
    xs: np.ndarray,
    ys: np.ndarray,
    poly: Polygon,
    points: Optional[np.ndarray] = None,
    tol_m: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # Drop NaNs to avoid invalid geometries
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return xs, ys

    spline_line = LineString(np.column_stack([xs, ys]))
    try:
        clipped = spline_line.intersection(poly)
    except Exception:
        return xs, ys

    if clipped.is_empty:
        return xs, ys

    def geom_xy(g):
        xg, yg = g.xy
        return np.asarray(xg), np.asarray(yg)

    if clipped.geom_type == "MultiLineString":
        geoms = list(clipped.geoms)
        if points is not None and len(points) > 0:
            pts = np.asarray(points)
            def score(g):
                # count wells that lie within tol_m of this subpath
                return sum(g.distance(Point(px, py)) <= tol_m for px, py in pts)
            scores = np.array([score(g) for g in geoms], dtype=int)
            max_score = int(scores.max())
            cands = [g for g, s in zip(geoms, scores) if s == max_score]
            geom = max(cands, key=lambda g: g.length)
        else:
            geom = max(geoms, key=lambda g: g.length)
        xs_c, ys_c = geom_xy(geom)
    else:
        xs_c, ys_c = geom_xy(clipped)

    return np.asarray(xs_c), np.asarray(ys_c)


def inside_all(xs: Sequence[float], ys: Sequence[float], poly: Polygon) -> bool:
    return all(poly.covers(Point(px, py)) for px, py in zip(xs, ys))


def _order_points_along_path(xs: np.ndarray, ys: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return points sorted by their nearest index along the sampled path."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(points)
    if len(xs) == 0 or len(pts) == 0:
        return pts
    idxs = [int(np.argmin((xs - px) ** 2 + (ys - py) ** 2)) for px, py in pts]
    order = np.argsort(np.asarray(idxs))
    return pts[order]


def point_to_path_distances(xs: np.ndarray, ys: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from each point to its nearest path sample."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(points)
    if len(xs) == 0 or len(pts) == 0:
        return np.full(len(pts), np.nan)
    dists = [float(np.sqrt(np.min((xs - px) ** 2 + (ys - py) ** 2))) for px, py in pts]
    return np.asarray(dists, dtype=float)

def force_path_through_wells(xs: np.ndarray, ys: np.ndarray, points: np.ndarray, keep_tails: bool = True) -> np.ndarray:
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = _order_points_along_path(xs, ys, np.asarray(points))

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) == 0 or len(pts) == 0:
        return pts

    idxs = []
    for wx, wy in pts:
        j = int(np.argmin((xs - wx) ** 2 + (ys - wy) ** 2))
        xs[j], ys[j] = wx, wy
        idxs.append(j)

    out = []

    # Optional: keep the leading tail before the first well
    if keep_tails and len(idxs) > 0 and idxs[0] > 0:
        lead = np.column_stack([xs[:idxs[0] + 1], ys[:idxs[0] + 1]])
        out.append(lead)

    # Segments between consecutive wells
    for k in range(len(idxs) - 1):
        a, b = idxs[k], idxs[k + 1]
        if a <= b:
            seg = np.column_stack([xs[a:b + 1], ys[a:b + 1]])
        else:
            seg = np.column_stack([xs[b:a + 1][::-1], ys[b:a + 1][::-1]])
        if len(out) > 0 and len(seg) > 0 and np.allclose(out[-1][-1], seg[0]):
            seg = seg[1:]
        out.append(seg)

    # Optional: keep the trailing tail after the last well
    if keep_tails and len(idxs) > 0 and idxs[-1] < len(xs) - 1:
        tail = np.column_stack([xs[idxs[-1]:], ys[idxs[-1]:]])
        if len(out) > 0 and len(tail) > 0 and np.allclose(out[-1][-1], tail[0]):
            tail = tail[1:]
        out.append(tail)

    if not out:
        return np.column_stack([xs, ys])
    return np.vstack(out)
