# app_config/__init__.py
import yaml
from pathlib import Path

# 1) Where am I installed?
PACKAGE_DIR  = Path(__file__).parent
# 2) Project root is _one_ level up
PROJECT_ROOT = PACKAGE_DIR.parent

# 3) Load YAML once
with open(PACKAGE_DIR / "config.yaml", "r") as f:
    _raw = yaml.safe_load(f)

# 4) Turn `data/...` into absolute Paths
for key, rel in _raw["data"].items():
    _raw["data"][key] = (PROJECT_ROOT / rel).resolve()

cfg = _raw
