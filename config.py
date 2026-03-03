import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict

import yaml


@dataclass
class PathsConfig:
    depth_segy_path: str
    wells_csv_path: str
    velocity_cube_path: str
    # plots_boundary_dir: str
    # plots_downsampled_seis_dir: str
    # plots_seis_well_tie_dir: str
    h5_output_dir: str
    las_paths: Optional[Dict[str, str]] = None
    figures_dir: str = "figures"


@dataclass
class ProcessConfig:
    dataset_name: str = "Cube1"
    max_depth_m: float = 3200.0
    resample_nz: int = 256
    resample_nx: int = 512
    num_splines_to_process: int = 2
    ricker_duration_s: float = 0.128
    ricker_dt_s: float = 0.001
    ricker_f0_hz: float = 25.0
    spline_orders: Optional[Dict[int, int]] = None  # e.g., {3: 5, 4: 18, 9: 4}
    spline_extension_seed: Optional[int] = None     # RNG seed for random endpoint extension
    spline_extension_max_m: float = 0.0             # Max extension (meters) per end (0=disabled)


@dataclass
class Config:
    paths: PathsConfig
    process: ProcessConfig


def load_config(config_path: str | os.PathLike) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    paths = raw.get("paths", {})
    process = raw.get("process", {})

    cfg = Config(
        paths=PathsConfig(
            depth_segy_path=paths["depth_segy_path"],
            wells_csv_path=paths["wells_csv_path"],
            velocity_cube_path=paths["velocity_cube_path"],
            # plots_boundary_dir=paths["plots_boundary_dir"],
            # plots_downsampled_seis_dir=paths["plots_downsampled_seis_dir"],
            # plots_seis_well_tie_dir=paths["plots_seis_well_tie_dir"],
            h5_output_dir=paths["h5_output_dir"],
            las_paths=paths.get("las_paths"),
            figures_dir=paths.get("figures_dir", "figures"),
        ),
        process=ProcessConfig(
            dataset_name=process.get("dataset_name"),
            max_depth_m=float(process.get("max_depth_m", 3200.0)),
            resample_nz=int(process.get("resample_nz", 256)),
            resample_nx=int(process.get("resample_nx", 512)),
            num_splines_to_process=int(process.get("num_splines_to_process", 2)),
            ricker_duration_s=float(process.get("ricker_duration_s", 0.128)),
            ricker_dt_s=float(process.get("ricker_dt_s", 0.001)),
            ricker_f0_hz=float(process.get("ricker_f0_hz", 25.0)),
            spline_orders={int(k): int(v) for k, v in process.get("spline_orders", {}).items()} if isinstance(process.get("spline_orders"), dict) else None,
            spline_extension_seed=process.get("spline_extension_seed"),
            spline_extension_max_m=float(process.get("spline_extension_max_m", 0.0)),
        ),
    )

    # Normalize to absolute paths
    for field in [
        "depth_segy_path",
        "wells_csv_path",
        "velocity_cube_path",
        # "plots_boundary_dir",
        # "plots_downsampled_seis_dir",
        # "plots_seis_well_tie_dir",
        "h5_output_dir",
        "figures_dir",
    ]:
        value = getattr(cfg.paths, field)
        setattr(cfg.paths, field, str(pathlib.Path(value).resolve()))

    return cfg


