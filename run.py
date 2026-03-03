import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline import run_3wells_pipeline


if __name__ == "__main__":
    base_cfg_dir = pathlib.Path(__file__).resolve().parent / "config"
    for cube_idx in range(1, 7):
        cfg_path = base_cfg_dir / f"ukndr_cube{cube_idx}.yaml"
        print(f"\n=== Running pipeline for Cube {cube_idx}: {cfg_path} ===")
        try:
            run_3wells_pipeline(str(cfg_path))
        except Exception as e:
            print(f"[WARN] Cube {cube_idx} failed: {e}")
