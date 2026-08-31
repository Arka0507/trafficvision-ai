from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.processing.pipeline import VideoProcessor
from backend.schemas import ProcessingOptions


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one video without starting the web interface.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "cli-job")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-car-model", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")

    options = ProcessingOptions(
        confidence=args.confidence,
        horizontal_fov_degrees=args.fov,
        device=args.device,
        enable_vehicle_classifier=not args.no_car_model,
    )
    processor = VideoProcessor(options)

    def progress(value: float, phase: str, message: str, stats: dict) -> None:
        print(f"[{value:6.2f}%] {phase}: {message}", flush=True)

    result = processor.process(args.video.resolve(), args.output.resolve(), progress, lambda: False)
    print(json.dumps(result["summary"]["results"], indent=2))
    print(f"Output: {result['artifacts']['video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
