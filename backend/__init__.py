"""TrafficVision AI backend package."""

import os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(_project_root / "data" / ".ultralytics"))

__version__ = "1.0.0"
