from typing import Tuple

import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import resample


def interp_line(ds: xr.Dataset, coords_xy: np.ndarray, bin_spacing_hint: int = 10) -> xr.Dataset:
    return ds.segysak.interp_line(coords_xy, bin_spacing_hint=bin_spacing_hint)


def crop_depth(seis: np.ndarray, depth: np.ndarray, max_depth_m: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = depth <= max_depth_m
    return seis[mask, :], depth[mask]


def normalize_symmetric(seis: np.ndarray, percentile: float = 90.5) -> Tuple[np.ndarray, TwoSlopeNorm]:
    scale = np.nanmax(np.abs(seis))
    seis_scaled = seis / (scale + 1e-9)
    amax = np.percentile(np.abs(seis_scaled), percentile)
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0, vmax=amax)
    return seis_scaled, norm


def resample_section(seis: np.ndarray, depth: np.ndarray, out_nz: int = 256, out_nx: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seis_z = resample(seis, out_nz, axis=0)
    depth_resampled = np.linspace(depth.min(), depth.max(), out_nz)
    seis_znx = resample(seis_z, out_nx, axis=1)
    x_axis_resampled = np.linspace(0, 1, out_nx)  # placeholder when physical x is not required
    return seis_znx, depth_resampled, x_axis_resampled


def _cosine_highfreq_taper_rfft_length(nz: int, start_frac: float, end_frac: float, taper_power: float = 2.0) -> np.ndarray:
    """Build a 1D raised-cosine taper in the rFFT frequency domain (length nz//2+1).
    - start_frac: normalized frequency (0..1) where taper begins (gain starts to drop)
    - end_frac: normalized frequency (0..1) where taper reaches zero (fully attenuated)
    - taper_power: >1 makes the roll-off more aggressive (steeper attenuation)
    """
    # Guard rails
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    end_frac = float(np.clip(end_frac, 0.0, 1.0))
    if end_frac <= start_frac:
        end_frac = min(1.0, start_frac + 0.1)
    taper_power = max(1.0, float(taper_power))

    # Normalized frequency grid for rfft bins
    freqs = np.fft.rfftfreq(nz, d=1.0)  # 0..0.5 (cycles/sample) if d=1
    fmax = freqs.max() if freqs.size else 1.0
    fr = freqs / (fmax if fmax > 0 else 1.0)  # normalize to 0..1

    # Piecewise 1 (passband) -> cosine rolloff -> 0 (stopband)
    w = np.ones_like(fr)
    roll = (fr - start_frac) / max(1e-12, (end_frac - start_frac))
    mask_roll = (fr >= start_frac) & (fr <= end_frac)
    w[fr >= end_frac] = 0.0
    base = 0.5 * (1.0 + np.cos(np.pi * roll[mask_roll]))  # 1 at start_frac → 0 at end_frac
    w[mask_roll] = np.power(np.clip(base, 0.0, 1.0), taper_power)
    return w


def resample_section_freq_taper_z(
    seis: np.ndarray,
    depth: np.ndarray,
    out_nz: int = 256,
    out_nx: int = 512,
    taper_start_frac: float = 0.6,
    taper_end_frac: float = 0.9,
    taper_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample a seismic section with frequency-domain taper on Z (depth/time) axis only.

    - Z (axis=0): apply raised-cosine taper in rFFT magnitude to smoothly suppress high (or mid/high) frequencies,
      then downsample to out_nz using FFT-based resample.
    - X (axis=1): classic FFT resample to out_nx (no spectral taper).

    Parameters
    - taper_start_frac: normalized frequency (0..1) where tapering begins (1.0 = near Nyquist)
    - taper_end_frac: normalized frequency (0..1) where gain reaches 0 (must be > start)
    """
    if seis.ndim != 2:
        raise ValueError("seis must be 2D array shaped (nz, nx)")
    nz, nx = seis.shape
    if nz < 4:
        raise ValueError("seis has too few samples along z to apply spectral taper")

    # Build taper window for rFFT bins along z
    taper = _cosine_highfreq_taper_rfft_length(nz, taper_start_frac, taper_end_frac, taper_power=taper_power)  # shape (nz//2+1,)

    # Apply along each trace (column)
    spec = np.fft.rfft(seis, axis=0)                       # (nz//2+1, nx)
    spec_tapered = spec * taper[:, None]                   # broadcast taper across traces
    seis_tapered = np.fft.irfft(spec_tapered, n=nz, axis=0).astype(seis.dtype, copy=False)

    # Downsample: Z then X
    seis_z = resample(seis_tapered, out_nz, axis=0)
    depth_resampled = np.linspace(depth.min(), depth.max(), out_nz)
    seis_znx = resample(seis_z, out_nx, axis=1)
    return seis_znx, depth_resampled


def resample_section_2d_fft_lowpass(
    seis: np.ndarray,
    depth: np.ndarray,
    out_nz: int,
    out_nx: int,
    edge_taper_frac: float = 0.1,
    taper_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D-FFT downsampling with smooth low-pass to avoid aliasing.

    - Takes 2D FFT, applies a 2D raised-cosine (cosine^taper_power) taper
      towards the new cutoff defined by the target size, crops the centered
      spectrum to (out_nz, out_nx), and iFFT back.

    Args:
      seis: (nz, nx) input section
      depth: (nz,) depth vector for input
      out_nz, out_nx: target size
      edge_taper_frac: last fraction (near cutoff) to smoothly roll to 0
      taper_power: >1 makes the roll-off steeper

    Returns:
      (out_nz, out_nx) resampled section and depth_resampled
    """
    if seis.ndim != 2:
        raise ValueError("seis must be 2D (nz, nx)")
    nz, nx = seis.shape
    if out_nz < 2 or out_nx < 2:
        raise ValueError("out_nz/out_nx must be >= 2")

    # FFT, center spectrum
    a = np.nan_to_num(seis, nan=0.0)
    spec = np.fft.fftshift(np.fft.fft2(a))

    # Build smooth 2D taper to the new cutoff envelope
    cz, cx = nz // 2, nx // 2
    hz_keep = max(out_nz / 2.0, 1e-9)
    hx_keep = max(out_nx / 2.0, 1e-9)

    kz = np.abs(np.arange(nz) - cz)[:, None] / hz_keep  # (nz, 1)
    kx = np.abs(np.arange(nx) - cx)[None, :] / hx_keep  # (1, nx)
    r = np.maximum(kz, kx)  # elliptical radius relative to new passband

    start = float(max(0.0, 1.0 - edge_taper_frac))
    end = 1.0

    mask = np.ones((nz, nx), dtype=float)
    mask[r > end] = 0.0
    roll = (r - start) / max(1e-12, (end - start))
    roll_mask = (r >= start) & (r <= end)
    base = 0.5 * (1.0 + np.cos(np.pi * np.clip(roll[roll_mask], 0.0, 1.0)))
    mask[roll_mask] = np.power(base, max(1.0, float(taper_power)))

    spec_lp = spec * mask

    # Crop centered spectrum to target size
    z0 = int(np.floor(cz - hz_keep))
    x0 = int(np.floor(cx - hx_keep))
    z1 = z0 + out_nz
    x1 = x0 + out_nx
    spec_crop = spec_lp[z0:z1, x0:x1]

    # Inverse FFT back to space, centered
    out = np.fft.ifft2(np.fft.ifftshift(spec_crop)).real
    depth_resampled = np.linspace(depth.min(), depth.max(), out_nz)

    return out.astype(seis.dtype, copy=False), depth_resampled