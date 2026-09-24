from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    path = Path(path or ROOT / "config.yaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = str(ROOT)
    return cfg
