# Distance and speed calibration

## Default FOV mode

FOV mode needs only the video's width and the camera's horizontal field of view. It estimates distance using a typical physical height for the detected class and the bounding-box pixel height. This works as a rough visual indicator, but bounding-box jitter, object pose, partial occlusion, and non-standard object size cause error.

The application marks FOV distance and speed with `~` and labels the video `MONOCULAR ESTIMATE`.

Use the **Distance scale** control to correct a consistent range bias:

1. Place or identify an object at a known distance.
2. Run a short clip.
3. Compute `known distance / displayed distance`.
4. Enter that ratio as Distance scale.

Example: known range is 30 m and displayed range is 24 m. Use `30 / 24 = 1.25`.

## Recommended four-point road-plane mode

Road-plane mode transforms the bottom-centre of each bounding box from image pixels to measured ground coordinates in metres.

### Collect the four image points

Choose four visible points on the road surface that form a large quadrilateral, ideally covering the travel area. Read their pixel coordinates `(x, y)` from the original video frame. Enter them in the same clockwise order, for example:

```json
[[420, 900], [1500, 900], [1080, 510], [840, 510]]
```

### Measure matching ground points

Choose an origin on the road and measure each selected point in metres as `(lateral_x, forward_y)`. Enter the matching points in exactly the same order:

```json
[[-6, 0], [6, 0], [3, 45], [-3, 45]]
```

The numbers shown above are format examples only; do not use them without measuring your scene.

### Validate

1. Process a short clip.
2. Check stationary road features and known vehicle travel times.
3. Compare at least one result with a known distance and a known speed source.
4. Adjust Speed scale only after the road-plane coordinates are correct.

## Camera requirements

- Fixed camera: recommended and assumed by camera-relative speed.
- Minimal vibration and rolling-shutter distortion.
- Clear view of the object-ground contact point.
- Original frame rate retained.
- No digital zoom change during the clip.

For a moving dashcam, absolute object speed requires ego-motion or vehicle telemetry. This project reports motion relative to the camera and should not be presented as absolute road speed.

## Safety statement

Do not use unvalidated estimates for traffic enforcement, collision avoidance, medical decisions, or other safety-critical actions. Camera calibration and target-domain validation are required.
