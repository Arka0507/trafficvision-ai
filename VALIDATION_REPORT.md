# Validation report

Validation date: 2026-08-31

## End-to-end source-video run

The final pipeline was run against the complete user-supplied `Main_Traffic_Video.mp4`, not only a sample clip.

| Item | Verified value |
| --- | --- |
| Input dimensions | 1920 × 1080 |
| Input frame rate | 59.9401 FPS |
| Frames processed | 2,802 |
| Input/output duration | 46.747 seconds |
| Detector | `yolov8n.pt` at 640 px, confidence 0.30 |
| Tracker | ByteTrack with `config/bytetrack_traffic.yaml` |
| Car classifier | `twincar-group2/twincar-classifier` |
| Measurement mode | Monocular FOV estimate, 70° horizontal FOV |
| Unique tracks | 239 |
| Vehicle tracks | 121 |
| Frame-level detection rows | 48,789 plus CSV header |
| Track rows | 239 plus CSV header |
| Annotated media | H.264, 1920 × 1080, 2,802 frames |
| CPU processing rate | 9.93 FPS for the tested environment |

Unique track appearances by detected class were: 101 person, 99 car, 18 bicycle, 15 motorcycle, 11 bus, 11 truck, 6 traffic light, and 1 handbag. A ByteTrack ID can occasionally receive different YOLO classes during its lifetime, so per-class unique counts are not expected to sum to the global unique-track count.

## Output verification

- OpenCV read and processed every source frame.
- The annotated output was transcoded to browser-compatible H.264/YUV420p.
- `ffprobe` verified 1920 × 1080, 59.94 FPS, 46.746747 seconds, and 2,802 readable video frames.
- Early, middle, and late output frames were visually inspected for bounding boxes, track IDs, labels, trails, distance, speed, HUD state, and the uncalibrated-measurement notice.
- Track CSV, optional frame-detection CSV, and JSON summary were generated successfully.
- The FastAPI health endpoint, static UI route, and job-list endpoint returned valid HTTP responses from a live local server.

## Automated checks

```text
ruff check .                 pass
pytest -q                    7 passed
node --check frontend/app.js pass
python -m compileall ...     pass
```

## How to interpret the result

- The detector covers every class known to the selected YOLO weights; the default COCO weights contain 80 classes and cannot detect literally every real-world object category.
- Values prefixed by `~` are monocular estimates. They must not be interpreted as survey-grade distance or enforcement-grade speed.
- The tested source uses a fixed camera, which is appropriate for this pipeline. For metric road speed, configure the supplied four-point road-plane calibration using measured ground coordinates.
- Car make/model is a closed-set secondary classifier. It knows 196 Stanford Cars categories, not every vehicle sold worldwide. It requires repeated agreement and otherwise returns `Uncertain`; a high-confidence label can still be wrong on out-of-domain traffic.
- In dense scenes, on-video labels can overlap. The machine-readable reports preserve the complete per-frame data without that presentation constraint.
