from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE_PATH: Path =  PROJECT_ROOT / "configs/config.yaml"
PARAMS_FILE_PATH: Path = PROJECT_ROOT / "params.yaml"