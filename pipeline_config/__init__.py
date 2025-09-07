# pipeline_config/__init__.py
import yaml
from pathlib import Path
import os 

PACKAGE_DIR  = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# override via environment variable 
CONFIG_PATH = os.environ.get("CR_CONFIG_PATH", PACKAGE_DIR / "config.yaml")

# load YAML
with open(PACKAGE_DIR / "config.yaml", "r",  encoding="utf-8") as f:
    _raw = yaml.safe_load(f)

# resolve data paths to absolute Paths
for key, rel in _raw["data"].items():
    _raw["data"][key] = (PROJECT_ROOT / rel).resolve()

# resolve model output_dir to absolute Path
if "model" in _raw and "output_dir" in _raw["model"]:
    _raw["model"]["output_dir"] = (PROJECT_ROOT / _raw["model"]["output_dir"]).resolve()

# resolve model prediction_dir to absolute Path
if "model" in _raw and "prediction_dir" in _raw["model"]:
    _raw["model"]["prediction_dir"] = (PROJECT_ROOT / _raw["model"]["prediction_dir"]).resolve()

cfg = _raw
