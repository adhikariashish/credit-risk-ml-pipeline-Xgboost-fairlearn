# pipeline_config/__init__.py
import yaml
from pathlib import Path

PACKAGE_DIR  = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# load YAML
with open(PACKAGE_DIR / "config.yaml", "r") as f:
    _raw = yaml.safe_load(f)

# resolve data paths to absolute Paths
for key, rel in _raw["data"].items():
    _raw["data"][key] = (PROJECT_ROOT / rel).resolve()

cfg = _raw
